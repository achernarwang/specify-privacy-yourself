import datasets
from tqdm import trange
from typing import Optional, Union, Dict
from transformers import (
    AutoProcessor, GenerationConfig, TrainingArguments,
    Trainer, TrainerControl, TrainerState, TrainerCallback
)
from torch.utils.data import Dataset
from accelerate.utils import gather_object, is_wandb_available
from trl.models.utils import unwrap_model_for_generation

if is_wandb_available():
    import wandb
    import pandas as pd

# Modified from trl LogCompletionsCallback
class LogGenerationsCallback(TrainerCallback):
    def __init__(
        self,
        trainer: Trainer,
        generation_config: Optional[GenerationConfig] = None,
        num_prompts: Optional[int] = None,
        freq: Optional[int] = None,
        processor: Optional[AutoProcessor] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
    ):
        self.trainer = trainer
        self.generation_config = generation_config
        self.freq = freq
        self.processor = processor
        self.table = []
        self._last_logged_step = -1

        if eval_dataset is None:
            raise ValueError("Trainer must have an evaluation dataset to use the LogCompletionsCallback.")
        else:
            self.eval_dataset = eval_dataset

        if num_prompts is not None:
            self.eval_dataset = self.eval_dataset.select(range(num_prompts))

    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Only log once per step (this method may be called multiple times)
        if state.global_step == self._last_logged_step:
            return

        # Only log every `freq` steps (if no `freq` is provided, log every `eval_steps` steps)
        freq = self.freq or state.eval_steps
        if state.global_step % freq != 0:
            return

        accelerator = self.trainer.accelerator
        model = self.trainer.model_wrapped
        batch_size = args.per_device_eval_batch_size
        with accelerator.split_between_processes(self.eval_dataset) as eval_dataset:
            input_messages = [msg[:1] for msg in eval_dataset["messages"]]
            questions = [msg[0]["content"][1]["text"] for msg in eval_dataset["messages"]]
            exp_resps = [msg[1]["content"][0]["text"] for msg in eval_dataset["messages"]]
            labels = eval_dataset["p_labels"]
            input_images = eval_dataset["images"]
            input_texts = [self.processor.apply_chat_template(message, add_generation_prompt=True) for message in input_messages]

            completions = []
            with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
                for idx in trange(0, len(input_texts), batch_size):                    
                    input_batch = self.processor(
                        input_images[idx : idx + batch_size],
                        input_texts[idx : idx + batch_size],
                        return_tensors="pt",
                        padding=True,
                        padding_side="left",
                    ).to(model.device)
                    
                    generations = unwrapped_model.generate(
                        **input_batch,
                        generation_config=self.generation_config,
                    )

                    output_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_batch.input_ids, generations)]
                    completions.extend(self.processor.batch_decode(output_ids_trimmed, skip_special_tokens=True))
            
            completions = gather_object(completions)
            questions = gather_object(questions)
            exp_resps = gather_object(exp_resps)
            labels = gather_object(labels)

        # Build the data to log
        global_step = [str(state.global_step)] * len(completions)
        data = list(zip(global_step, labels, questions, exp_resps, completions))
        self.table.extend(data)

        if "wandb" in args.report_to and is_wandb_available():
            if self.trainer.accelerator.is_main_process:
                table = pd.DataFrame(columns=["step", "label", "question", "ground truth", "completion"], data=self.table)
                wandb.log({"completions": table})

        # Strong coupling with eval dataset
        if "tensorboard" in args.report_to:
            # group by label
            data_by_label = {"eval_" + l: [] for l in set(labels)}
            for d in data:
                data_by_label["eval_" + d[1]].append(f"**Question**: {d[2]!r}\n**Ground Truth**: {d[3]!r}\n**Completion**: {d[4]!r}")
            for label, content in data_by_label.items():
                data_by_label[label] = "\n\n".join(content)

            should_log = control.should_log
            self.trainer.log(data_by_label)
            control.should_log = should_log

        # Save the last logged step, so we don't log the same completions multiple times
        self._last_logged_step = state.global_step

        return control