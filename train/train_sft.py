import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, GenerationConfig, Qwen2VLForConditionalGeneration
from liger_kernel.transformers import monkey_patch
from accelerate.utils import is_wandb_available

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from params import ScriptArguments
from data_utils import prepare_dataset
from log_utils import LogGenerationsCallback

if is_wandb_available():
    import wandb


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    ################
    # Model, Tokenizer & Processor
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
    )

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, min_pixels=script_args.min_pixels, max_pixels=script_args.max_pixels
    )

    if training_args.use_liger:
        print("Applying Liger Kernel to Qwen2-VL model")
        monkey_patch.apply_liger_kernel_to_qwen2_vl(
            # These args can be used to override the default Liger settings
            # cross_entropy=True,
            # fused_linear_cross_entropy=False,
        )

    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )
    assert isinstance(model, Qwen2VLForConditionalGeneration), "Currently only Qwen2VL model is supported."

    ################
    # Create a data collator to encode text and image pairs
    ################
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        images = [example["images"] for example in examples]

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True, add_special_tokens=False)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  #
        # Ignore the image token index in the loss computation (model specific)
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        labels[labels == image_token_id] = -100

        batch["labels"] = labels

        return batch

    ################
    # Dataset
    ################
    train_set = prepare_dataset(script_args.train_data_path, script_args.label_path, script_args.image_folder, script_args.add_distractors, None, script_args.shuffle, training_args.seed)
    # train_set.shuffle(seed=training_args.seed)
    if script_args.eval_data_path is None:
        eval_set = None
    else:
        eval_set = prepare_dataset(script_args.eval_data_path, script_args.label_path, script_args.image_folder, script_args.add_distractors)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_set,
        eval_dataset=eval_set,
        processing_class=processor,
        peft_config=get_peft_config(model_args),
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
        generation_callback = LogGenerationsCallback(trainer, generation_config, script_args.num_eval_samples, processor=processor, eval_dataset=eval_set)
        trainer.add_callback(generation_callback)

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    
    if "wandb" in training_args.report_to and is_wandb_available():
        wandb.finish()
