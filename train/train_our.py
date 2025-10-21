import torch
from accelerate.utils import is_wandb_available
from liger_kernel.transformers import monkey_patch
from transformers import AutoModelForVision2Seq, AutoProcessor, GenerationConfig, Qwen2VLForConditionalGeneration
from trl import (
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from log_utils import LogGenerationsCallback
from params import ScriptArguments, CustomDPOConfig
from data_utils import prepare_dataset, prepare_our_dataset
from our_qwen2vl import Qwen2VLOurTrainer, Qwen2VLDataCollatorForOur

if is_wandb_available():
    import wandb


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, CustomDPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.remove_unused_columns = False
    # training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False
    )

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, min_pixels=script_args.min_pixels, max_pixels=script_args.max_pixels
    )
    tokenizer = processor.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad token to {tokenizer.eos_token}")

    if training_args.use_liger:
        print("Applying Liger Kernel to Qwen2-VL model")
        monkey_patch.apply_liger_kernel_to_qwen2_vl(
            # These args can be used to override the default Liger settings
            # cross_entropy=True,
            fused_linear_cross_entropy=False,
        )
    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )

    assert isinstance(model, Qwen2VLForConditionalGeneration), "Currently only Qwen2VL model is supported."

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_model = AutoModelForVision2Seq.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
    else:
        ref_model = None
    
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Patching for Qwen2VL
    ################
    data_collator = None
    if isinstance(model, Qwen2VLForConditionalGeneration):
        data_collator = Qwen2VLDataCollatorForOur(processor, training_args)
    
    ################
    # Dataset
    ################
    train_set = prepare_our_dataset(script_args.train_data_path, script_args.label_path, script_args.image_folder, script_args.add_distractors)
    if script_args.num_train_samples is not None:
        train_set = train_set.select(range(script_args.num_train_samples))
    
    gen_set = None
    if script_args.eval_data_path is not None:
        gen_set = prepare_dataset(script_args.eval_data_path, script_args.label_path, script_args.image_folder, script_args.add_distractors)

    ################
    # Training
    ################
    trainer = Qwen2VLOurTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=train_set,
        data_collator=data_collator,
        processing_class=processor,
        peft_config=peft_config,
    )

    if "wandb" in training_args.report_to and is_wandb_available():
        wandb.init(
            project=script_args.project,
            name=script_args.train_name,
            mode=script_args.wandb_mode,
            config=training_args,
        )

    if script_args.eval_data_path is not None:
        generation_config = GenerationConfig(max_new_tokens=script_args.eval_max_new_tokens, do_sample=False)
        generation_callback = LogGenerationsCallback(trainer, generation_config, script_args.num_eval_samples, processor=processor, eval_dataset=gen_set)
        trainer.add_callback(generation_callback)

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    if "wandb" in training_args.report_to and is_wandb_available():
        wandb.finish()