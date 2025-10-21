from contextlib import nullcontext
from typing import Any, Union, Literal

import torch
import torch.amp as amp
from torch import nn
import torch.nn.functional as F

from transformers import PreTrainedTokenizer
from transformers.utils import is_torch_xpu_available
from transformers.data.data_collator import DataCollatorMixin
from trl import DPOTrainer
from trl.trainer.dpo_config import FDivergenceConstants, FDivergenceType
from trl.trainer.utils import pad, pad_to_length, selective_log_softmax, flush_left, cap_exp


def apply_chat_template(example: dict[str, list[dict[str, str]]], tokenizer: PreTrainedTokenizer) -> dict[str, str]:
    last_role = example["prompt_0"][-1]["role"]
    if last_role == "user":
        add_generation_prompt = True
        continue_final_message = False
    elif last_role == "assistant":
        add_generation_prompt = False
        continue_final_message = True
    else:
        raise ValueError(f"Invalid role in the last message: {last_role}")
    
    prompt_0 = tokenizer.apply_chat_template(example["prompt_0"], tokenize=False, add_generation_prompt=add_generation_prompt, continue_final_message=continue_final_message)
    prompt_1 = tokenizer.apply_chat_template(example["prompt_1"], tokenize=False, add_generation_prompt=add_generation_prompt, continue_final_message=continue_final_message)
    
    prompt_chosen_0 = tokenizer.apply_chat_template(example["prompt_0"] + example["chosen_0"], tokenize=False)
    chosen_0 = prompt_chosen_0[len(prompt_0):]
    prompt_chosen_1 = tokenizer.apply_chat_template(example["prompt_1"] + example["chosen_1"], tokenize=False)
    chosen_1 = prompt_chosen_1[len(prompt_1):]
    prompt_rejected_0 = tokenizer.apply_chat_template(example["prompt_0"] + example["rejected_0"], tokenize=False)
    rejected_0 = prompt_rejected_0[len(prompt_0):]
    prompt_rejected_1 = tokenizer.apply_chat_template(example["prompt_1"] + example["rejected_1"], tokenize=False)
    rejected_1 = prompt_rejected_1[len(prompt_1):]
    
    return {
        "prompt_0": prompt_0,
        "prompt_1": prompt_1,
        "chosen_0": chosen_0,
        "chosen_1": chosen_1,
        "rejected_0": rejected_0,
        "rejected_1": rejected_1,
    }

class Qwen2VLDataCollatorForOur(DataCollatorMixin):
    def __init__(self, processor, args, return_tensors: str = "pt"):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.args = args
        self.return_tensors = return_tensors
        if self.args.padding_value is not None:
            self.pad_token_id = args.padding_value
        else:
            if hasattr(self.processor, "pad_token_id") and self.processor.pad_token_id is not None:
                self.pad_token_id = self.processor.pad_token_id
            elif self.tokenizer.pad_token_id is not None:
                self.pad_token_id = self.tokenizer.pad_token_id
            else:
                raise ValueError(
                    "`padding_value` is not specified in `DPOConfig`, and `pad_token_id` is missing in the "
                    "`processing_class`. Please either set the `padding_value` argument in `DPOConfig`, or set "
                    "`tokenizer.pad_token` (e.g., `tokenizer.pad_token = tokenizer.eos_token`) before instantiating "
                    "the trainer."
                )

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Applying chat template
        examples = [{**apply_chat_template(example, self.processor), "images": example["images"]} for example in examples]

        # Tokenizing
        processed_examples = []
        for features in examples:
            processed_features_0 = self.processor(images=features["images"], text=features["prompt_0"], add_special_tokens=False)
            prompt_input_ids_0 = processed_features_0["input_ids"][0]
            pixel_values = processed_features_0["pixel_values"]
            chosen_input_ids_0 = self.tokenizer(features["chosen_0"], add_special_tokens=False)["input_ids"]
            rejected_input_ids_0 = self.tokenizer(features["rejected_0"], add_special_tokens=False)["input_ids"]

            chosen_input_ids_0 = chosen_input_ids_0 + [self.tokenizer.eos_token_id]
            rejected_input_ids_0 = rejected_input_ids_0 + [self.tokenizer.eos_token_id]

            processed_features_1 = self.processor(images=features["images"], text=features["prompt_1"], add_special_tokens=False)
            prompt_input_ids_1 = processed_features_1["input_ids"][0]
            chosen_input_ids_1 = self.tokenizer(features["chosen_1"], add_special_tokens=False)["input_ids"]
            rejected_input_ids_1 = self.tokenizer(features["rejected_1"], add_special_tokens=False)["input_ids"]
            
            chosen_input_ids_1 = chosen_input_ids_1 + [self.tokenizer.eos_token_id]
            rejected_input_ids_1 = rejected_input_ids_1 + [self.tokenizer.eos_token_id]

            output = {
                "prompt_input_ids_0": prompt_input_ids_0,
                "prompt_input_ids_1": prompt_input_ids_1,
                "pixel_values": pixel_values,
                "chosen_input_ids_0": chosen_input_ids_0,
                "chosen_input_ids_1": chosen_input_ids_1,
                "rejected_input_ids_0": rejected_input_ids_0,
                "rejected_input_ids_1": rejected_input_ids_1,
            }

            if "pixel_attention_mask" in processed_features_0:
                output["pixel_attention_mask"] = processed_features_0["pixel_attention_mask"][0]
            if "image_sizes" in processed_features_0:
                output["image_sizes"] = processed_features_0["image_sizes"][0]
            if "image_grid_thw" in processed_features_0:
                output["image_grid_thw"] = processed_features_0["image_grid_thw"][0]
            
            processed_examples.append(output)

        examples = processed_examples
        # Convert to tensor
        prompt_input_ids_0 = [torch.tensor(example["prompt_input_ids_0"]) for example in examples]
        prompt_attention_mask_0 = [torch.ones_like(input_ids) for input_ids in prompt_input_ids_0]
        chosen_input_ids_0 = [torch.tensor(example["chosen_input_ids_0"]) for example in examples]
        chosen_attention_mask_0 = [torch.ones_like(input_ids) for input_ids in chosen_input_ids_0]
        rejected_input_ids_0 = [torch.tensor(example["rejected_input_ids_0"]) for example in examples]
        rejected_attention_mask_0 = [torch.ones_like(input_ids) for input_ids in rejected_input_ids_0]

        prompt_input_ids_1 = [torch.tensor(example["prompt_input_ids_1"]) for example in examples]
        prompt_attention_mask_1 = [torch.ones_like(input_ids) for input_ids in prompt_input_ids_1]
        chosen_input_ids_1 = [torch.tensor(example["chosen_input_ids_1"]) for example in examples]
        chosen_attention_mask_1 = [torch.ones_like(input_ids) for input_ids in chosen_input_ids_1]
        rejected_input_ids_1 = [torch.tensor(example["rejected_input_ids_1"]) for example in examples]
        rejected_attention_mask_1 = [torch.ones_like(input_ids) for input_ids in rejected_input_ids_1]
        
        if "pixel_values" in examples[0]:
            pixel_values = [torch.tensor(example["pixel_values"]) for example in examples]
        if "image_grid_thw" in examples[0]:
            image_grid_thw = [torch.tensor(example["image_grid_thw"]) for example in examples]
        if "pixel_attention_mask" in examples[0]:
            pixel_attention_mask = [torch.tensor(example["pixel_attention_mask"]) for example in examples]

        # Pad
        ### Optional: Another alternative implementation of our method is to concatenate the prompt_0 and prompt_1 here.
        output = {}
        output["prompt_input_ids_0"] = pad(prompt_input_ids_0, padding_value=self.pad_token_id, padding_side="left")
        output["prompt_attention_mask_0"] = pad(prompt_attention_mask_0, padding_value=0, padding_side="left")
        output["chosen_input_ids_0"] = pad(chosen_input_ids_0, padding_value=self.pad_token_id)
        output["chosen_attention_mask_0"] = pad(chosen_attention_mask_0, padding_value=0)
        output["rejected_input_ids_0"] = pad(rejected_input_ids_0, padding_value=self.pad_token_id)
        output["rejected_attention_mask_0"] = pad(rejected_attention_mask_0, padding_value=0)
        
        output["prompt_input_ids_1"] = pad(prompt_input_ids_1, padding_value=self.pad_token_id, padding_side="left")
        output["prompt_attention_mask_1"] = pad(prompt_attention_mask_1, padding_value=0, padding_side="left")
        output["chosen_input_ids_1"] = pad(chosen_input_ids_1, padding_value=self.pad_token_id)
        output["chosen_attention_mask_1"] = pad(chosen_attention_mask_1, padding_value=0)
        output["rejected_input_ids_1"] = pad(rejected_input_ids_1, padding_value=self.pad_token_id)
        output["rejected_attention_mask_1"] = pad(rejected_attention_mask_1, padding_value=0)
        
        if "pixel_values" in examples[0]:
            # output["pixel_values"] = pad(pixel_values, padding_value=0.0)
            output["pixel_values"] = torch.cat(pixel_values, dim=0)
        if "pixel_attention_mask" in examples[0]:
            output["pixel_attention_mask"] = pad(pixel_attention_mask, padding_value=0)
        if "image_sizes" in examples[0]:
            output["image_sizes"] = torch.tensor([example["image_sizes"] for example in examples])
        if "image_grid_thw" in examples[0]:
            output["image_grid_thw"] = torch.stack(image_grid_thw, dim=0)

        return output
    

class Qwen2VLOurTrainer(DPOTrainer):
    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        return dataset
    
    def compute_ref_log_probs(self, batch: dict[str, torch.LongTensor]) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        device_type = "xpu" if is_torch_xpu_available() else "cuda"
        compte_ref_context_manager = amp.autocast(device_type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        with torch.no_grad(), compte_ref_context_manager:
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_model_output = self.concatenated_forward(self.model, batch)
            else:
                ref_model_output = self.concatenated_forward(self.ref_model, batch)
        return ref_model_output["chosen_logps_0"], ref_model_output["chosen_logps_1"], ref_model_output["rejected_logps_0"], ref_model_output["rejected_logps_1"]
    
    
    @staticmethod
    def concatenated_inputs(
        batch: dict[str, Union[list, torch.LongTensor]], padding_value: int
    ) -> dict[str, torch.LongTensor]:
        """
        Concatenate the `chosen` and `rejected` inputs from the batch into a single tensor for both the prompt
        and completion sequences.

        Args:
            batch (`dict[str, Union[list, torch.LongTensor]]`):
                A batch of input data. The batch must contain the following keys:

                - `"prompt_input_ids"`: Tensor of shape `(batch_size, prompt_length)` representing the prompt input IDs.
                - `"chosen_input_ids"`: Tensor of shape `(batch_size, chosen_length)` representing the chosen completion input IDs.
                - `"rejected_input_ids"`: Tensor of shape `(batch_size, rejected_length)` representing the rejected completion input IDs.
                - `"prompt_pixel_values"` (optional): Tensor for pixel values, if available.
                - `"prompt_pixel_attention_mask"` (optional): Tensor for pixel attention masks, if available.

            padding_value (`int`):
                The padding value to use for the concatenated completion sequences (`chosen_input_ids` and
                `rejected_input_ids`).

        Returns:
            `dict[str, torch.LongTensor]`: A dictionary containing:

                - `"prompt_input_ids"`: Concatenated prompt input IDs of shape `(2 * batch_size, prompt_length)`.
                - `"completion_input_ids"`: Concatenated chosen and rejected completion input IDs of shape `(2 * batch_size, max_completion_length)`.
                - `"prompt_attention_mask"`: Concatenated prompt attention masks of shape `(2 * batch_size, prompt_length)`.
                - `"completion_attention_mask"`: Concatenated chosen and rejected attention masks of shape `(2 * batch_size, max_completion_length)`.
                - `"pixel_values"` (optional): Concatenated pixel values if `"prompt_pixel_values"` are present.
                - `"pixel_attention_mask"` (optional): Concatenated pixel attention masks if `"prompt_pixel_attention_mask"` are present.

        Notes:
            The completion input IDs and attention masks are padded to the maximum completion length of the chosen
            or rejected sequences.
        """
        output = {}

        if "pixel_values" in batch:
            output["pixel_values"] = torch.cat([batch["pixel_values"], batch["pixel_values"], batch["pixel_values"], batch["pixel_values"]], dim=0)

        if "pixel_attention_mask" in batch:
            output["pixel_attention_mask"] = torch.cat(
                [batch["pixel_attention_mask"], batch["pixel_attention_mask"], batch["pixel_attention_mask"], batch["pixel_attention_mask"]], dim=0
            )
        if "image_sizes" in batch:
            output["image_sizes"] = torch.cat([batch["image_sizes"], batch["image_sizes"], batch["image_sizes"], batch["image_sizes"]], dim=0)
        if "image_grid_thw" in batch:
            output["image_grid_thw"] = torch.cat([batch["image_grid_thw"], batch["image_grid_thw"], batch["image_grid_thw"], batch["image_grid_thw"]], dim=0)

        # Concatenate the chosen and rejected messages
        max_message_length = max(
            batch["prompt_input_ids_0"].shape[1] + batch["chosen_input_ids_0"].shape[1],
            batch["prompt_input_ids_0"].shape[1] + batch["rejected_input_ids_0"].shape[1],
            batch["prompt_input_ids_1"].shape[1] + batch["chosen_input_ids_1"].shape[1],
            batch["prompt_input_ids_1"].shape[1] + batch["rejected_input_ids_1"].shape[1]
        )
        
        
        output["message_input_ids"] = torch.cat(
            (
                pad_to_length(torch.cat((batch["prompt_input_ids_0"], batch["chosen_input_ids_0"]), dim=1), max_message_length, pad_value=padding_value),
                pad_to_length(torch.cat((batch["prompt_input_ids_1"], batch["chosen_input_ids_1"]), dim=1), max_message_length, pad_value=padding_value),
                pad_to_length(torch.cat((batch["prompt_input_ids_0"], batch["rejected_input_ids_0"]), dim=1), max_message_length, pad_value=padding_value),
                pad_to_length(torch.cat((batch["prompt_input_ids_1"], batch["rejected_input_ids_1"]), dim=1), max_message_length, pad_value=padding_value),
            ),
            dim=0
        )
        output["message_attention_mask"] = torch.cat(
            (
                pad_to_length(torch.cat((batch["prompt_attention_mask_0"], batch["chosen_attention_mask_0"]), dim=1), max_message_length, pad_value=0),
                pad_to_length(torch.cat((batch["prompt_attention_mask_1"], batch["chosen_attention_mask_1"]), dim=1), max_message_length, pad_value=0),
                pad_to_length(torch.cat((batch["prompt_attention_mask_0"], batch["rejected_attention_mask_0"]), dim=1), max_message_length, pad_value=0),
                pad_to_length(torch.cat((batch["prompt_attention_mask_1"], batch["rejected_attention_mask_1"]), dim=1), max_message_length, pad_value=0),
            ),
            dim=0
        )
        output["message_loss_mask"] = torch.cat(
            (
                pad_to_length(torch.cat((torch.zeros_like(batch["prompt_attention_mask_0"]), batch["chosen_attention_mask_0"]), dim=1), max_message_length, pad_value=0),
                pad_to_length(torch.cat((torch.zeros_like(batch["prompt_attention_mask_1"]), batch["chosen_attention_mask_1"]), dim=1), max_message_length, pad_value=0),
                pad_to_length(torch.cat((torch.zeros_like(batch["prompt_attention_mask_0"]), batch["rejected_attention_mask_0"]), dim=1), max_message_length, pad_value=0),
                pad_to_length(torch.cat((torch.zeros_like(batch["prompt_attention_mask_1"]), batch["rejected_attention_mask_1"]), dim=1), max_message_length, pad_value=0),
            ),
            dim=0
        )

        return output
    
    def concatenated_forward(self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]]):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        num_examples = batch["prompt_input_ids_0"].shape[0]

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # Add the pixel values and attention masks for vision models
        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "pixel_attention_mask" in concatenated_batch:
            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]
        if "image_sizes" in concatenated_batch:
            model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]
        if "image_grid_thw" in concatenated_batch:
            model_kwargs["image_grid_thw"] = concatenated_batch["image_grid_thw"]

        message_input_ids = concatenated_batch["message_input_ids"]
        message_attention_mask = concatenated_batch["message_attention_mask"]
        message_loss_mask = concatenated_batch["message_loss_mask"]
        
        assert self.is_encoder_decoder == False

        #########################################################
        # Concatenate the prompt and completion inputs
        input_ids = message_input_ids
        attention_mask = message_attention_mask
        # Mask the prompt but not the completion for the loss
        loss_mask = message_loss_mask
        # loss_mask = attention_mask.clone()

        # Flush left to reduce the memory usage
        # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
        #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
        attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

        # Truncate right
        if self.max_length is not None:
            if self.truncation_mode == "keep_end":
                input_ids = input_ids[:, -self.max_length :]
                attention_mask = attention_mask[:, -self.max_length :]
                loss_mask = loss_mask[:, -self.max_length :]
            elif self.truncation_mode == "keep_start":
                input_ids = input_ids[:, : self.max_length]
                attention_mask = attention_mask[:, : self.max_length]
                loss_mask = loss_mask[:, : self.max_length]
            else:
                raise ValueError(
                    f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', "
                    "'keep_start']."
                )

        if self.use_logits_to_keep:
            # Compute logits_to_keep based on loss_mask pattern:
            # [[0, 0, 0, x, x, x, x],
            #  [0, 0, 0, x, x, x, 0]]
            #         ^ start computing logits from here ([:, -(7-3+1):])
            first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
            logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1  # +1 for the first label
            model_kwargs["logits_to_keep"] = logits_to_keep

        if self.padding_free:
            # Flatten the input_ids, position_ids, and loss_mask
            # input_ids = [[a, b, c, 0], ->     input_ids = [[a, b, c, d, e, f, g]]
            #              [d, e, f, g]]     position_ids = [[0, 1, 2, 0, 1, 2, 3]]
            input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
            loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
            position_ids = attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
            model_kwargs["position_ids"] = position_ids
        else:
            model_kwargs["attention_mask"] = attention_mask

        outputs = model(input_ids, **model_kwargs)
        logits = outputs.logits

        # Offset the logits by one to align with the labels
        labels = torch.roll(input_ids, shifts=-1, dims=1)
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

        if self.use_logits_to_keep:
            # Align labels with logits
            # logits:    -,  -, [x2, x3, x4, x5, x6]
            #                     ^ --------- ^       after logits[:, :-1, :]
            # labels:   [y0, y1, y2, y3, y4, y5, y6]
            #                         ^ --------- ^   with logits_to_keep=4, [:, -4:]
            # loss_mask: [0,  0,  0,  1,  1,  1,  1]
            labels = labels[:, -logits_to_keep:]
            loss_mask = loss_mask[:, -logits_to_keep:]
        #########################################################

        if logits.shape[:2] != labels.shape[:2]:
            # for llava, the returned logits include the image tokens (placed before the text tokens)
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute the log probabilities of the labels
        labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        if self.padding_free:
            # Unflatten the per_token_logps (shape: [1, sum_seq_len] -> [batch_size, seq_len])
            batch_size, seq_len = attention_mask.shape
            per_token_logps_ = torch.zeros(
                batch_size, seq_len, device=outputs.logits.device, dtype=outputs.logits.dtype
            )
            per_token_logps_[attention_mask.bool()] = per_token_logps
            per_token_logps = per_token_logps_

        all_logps = per_token_logps.sum(-1)

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)  # same as sum(probs**2) in log space
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                chosen_weights_0 = all_weights[:num_examples]
                chosen_weights_1 = all_weights[num_examples:2*num_examples]
                rejected_weights_0 = all_weights[2*num_examples:3*num_examples]
                rejected_weights_1 = all_weights[3*num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights_0 + chosen_weights_1 + rejected_weights_0 + rejected_weights_1), max=1)

        if self.args.rpo_alpha is not None:
            # Only use the chosen logits for the RPO loss
            chosen_logits = logits[:2 * num_examples]
            chosen_labels = labels[:2 * num_examples]

            # Compute the log probabilities of the labels
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
            )

        if self.loss_type == "ipo":
            all_logps = all_logps / loss_mask.sum(-1)

        output["chosen_logps_0"] = all_logps[:num_examples]
        output["chosen_logps_1"] = all_logps[num_examples:2*num_examples]
        output["rejected_logps_0"] = all_logps[2*num_examples:3*num_examples]
        output["rejected_logps_1"] = all_logps[3*num_examples:]

        # Compute the mean logits
        if self.padding_free:
            # position_ids contains a sequence of range identifiers (e.g., [[0, 1, 2, 0, 1, 2, 3, ...]]).
            # There are 2*num_examples ranges in total: the first half corresponds to the chosen tokens,
            # and the second half to the rejected tokens.
            # To find the start of the rejected tokens, we look for the num_examples+1-th zero in pos_id.
            split_idx = (position_ids == 0).nonzero(as_tuple=True)[1][: 2*num_examples]
            mean_chosen_logits = logits[0, :split_idx][loss_mask[0, :split_idx]].mean()
            mean_rejected_logits = logits[0, split_idx:][loss_mask[0, split_idx:]].mean()
        else:
            mean_chosen_logits = logits[:2*num_examples][loss_mask[:2*num_examples]].mean()
            mean_rejected_logits = logits[2*num_examples:][loss_mask[2*num_examples:]].mean()

        output["mean_chosen_logits"] = mean_chosen_logits
        output["mean_rejected_logits"] = mean_rejected_logits

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output
    
    def dpo_loss(
        self,
        chosen_logps_0: torch.FloatTensor,
        chosen_logps_1: torch.FloatTensor,
        rejected_logps_0: torch.FloatTensor,
        rejected_logps_1: torch.FloatTensor,
        ref_chosen_logps_0: torch.FloatTensor,
        ref_chosen_logps_1: torch.FloatTensor,
        ref_rejected_logps_0: torch.FloatTensor,
        ref_rejected_logps_1: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            chosen_logps (`torch.FloatTensor`):
                Log probabilities of the model for the chosen responses. Shape: `(batch_size,)`.
            rejected_logps (`torch.FloatTensor`):
                Log probabilities of the model for the rejected responses. Shape: `(batch_size,)`.
            ref_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the chosen responses. Shape: `(batch_size,)`.
            ref_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the rejected responses. Shape: `(batch_size,)`.

        Returns:
            A tuple of three tensors: `(losses, chosen_rewards, rejected_rewards)`.
            The losses tensor contains the DPO loss for each example in the batch.
            The `chosen_rewards` and `rejected_rewards` tensors contain the rewards for the chosen and rejected
            responses, respectively.
        """
        device = self.accelerator.device

        # Get the log ratios for the chosen and rejected responses
        chosen_logratios_0 = chosen_logps_0.to(device) - (not self.reference_free) * ref_chosen_logps_0.to(device)
        chosen_logratios_1 = chosen_logps_1.to(device) - (not self.reference_free) * ref_chosen_logps_1.to(device)
        rejected_logratios_0 = rejected_logps_0.to(device) - (not self.reference_free) * ref_rejected_logps_0.to(device)
        rejected_logratios_1 = rejected_logps_1.to(device) - (not self.reference_free) * ref_rejected_logps_1.to(device)

        if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE.value:
            # The alpha-divergence formula: (1 - u^-alpha) / alpha
            # The divergence difference between the chosen and rejected sample is:
            #     (1 - u[w]^-alpha) / alpha - (1 - u[l]^-alpha) / alpha
            #        = (u[l]^-alpha - u[w]^-alpha) / alpha
            # where u[w] and u[l] are the policy/reference probability ratios
            # for the chosen and rejected samples, respectively.
            alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
            if self.f_divergence_params and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY in self.f_divergence_params:
                alpha_coef = float(self.f_divergence_params[FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY])
            logits = (cap_exp(rejected_logratios_0 * -alpha_coef) + cap_exp(rejected_logratios_1 * -alpha_coef) - cap_exp(chosen_logratios_0 * -alpha_coef) - cap_exp(chosen_logratios_1 * -alpha_coef)) / alpha_coef
        else:
            logratios = chosen_logps_0 + chosen_logps_1 - rejected_logps_0 - rejected_logps_1
            if self.reference_free:
                ref_logratios = torch.tensor([0], dtype=logratios.dtype, device=logratios.device)
            else:
                ref_logratios = ref_chosen_logps_0 + ref_chosen_logps_1 - ref_rejected_logps_0 - ref_rejected_logps_1

            logratios = logratios.to(self.accelerator.device)
            ref_logratios = ref_logratios.to(self.accelerator.device)
            logits = logratios - ref_logratios

            if self.f_divergence_type == FDivergenceType.JS_DIVERGENCE.value:
                # The js-divergence formula: log(2 * u / (1 + u))
                # The divergence difference between the chosen and rejected sample is:
                #     log(2 * u[w] / (1 + u[w])) - log(2 * u[l] / (1 + u[l]))
                #       = log(u[w]) - log(u[l]) - (log(1 + u[w]) - log(1 + u[l]))
                # where u[w] and u[l] are the policy/reference probability ratios
                # for the chosen and rejected samples, respectively.
                logits -= F.softplus(chosen_logratios_0) + F.softplus(chosen_logratios_1) - F.softplus(rejected_logratios_0) - F.softplus(rejected_logratios_1)

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the
        # labels and calculates a conservative DPO loss.
        if self.loss_type == "nca_priv":
            chosen_rewards_0 = (chosen_logps_0 - ref_chosen_logps_0) * self.beta
            chosen_rewards_1 = (chosen_logps_1 - ref_chosen_logps_1) * self.beta
            rejected_rewards_0 = (rejected_logps_0 - ref_rejected_logps_0) * self.beta
            rejected_rewards_1 = (rejected_logps_1 - ref_rejected_logps_1) * self.beta
            losses = (
                -F.logsigmoid(chosen_rewards_0)
                - 0.5 * F.logsigmoid(-chosen_rewards_0)
                - 0.5 * F.logsigmoid(-rejected_rewards_0)
                -F.logsigmoid(chosen_rewards_1)
                - 0.5 * F.logsigmoid(-chosen_rewards_1)
                - 0.5 * F.logsigmoid(-rejected_rewards_1)
            )

        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be 'nca_priv'"
            )

        chosen_rewards = self.beta * (chosen_logps_0.to(device) - ref_chosen_logps_0.to(device) + chosen_logps_1.to(device) - ref_chosen_logps_1.to(device)).detach()
        rejected_rewards = self.beta * (rejected_logps_0.to(device) - ref_rejected_logps_0.to(device) + rejected_logps_1.to(device) - ref_rejected_logps_1.to(device)).detach()

        chosen_rewards_0 = self.beta * (chosen_logps_0.to(device) - ref_chosen_logps_0.to(device)).detach()
        chosen_rewards_1 = self.beta * (chosen_logps_1.to(device) - ref_chosen_logps_1.to(device)).detach()
        rejected_rewards_0 = self.beta * (rejected_logps_0.to(device) - ref_rejected_logps_0.to(device)).detach()
        rejected_rewards_1 = self.beta * (rejected_logps_1.to(device) - ref_rejected_logps_1.to(device)).detach()

        return losses, chosen_rewards, rejected_rewards, chosen_rewards_0, chosen_rewards_1, rejected_rewards_0, rejected_rewards_1
    

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        model_output = self.concatenated_forward(model, batch)

        # if ref_chosen_logps and ref_rejected_logps in batch use them, otherwise use the reference model
        if "ref_chosen_logps_0" in batch and "ref_rejected_logps_0" in batch and "ref_chosen_logps_1" in batch and "ref_rejected_logps_1" in batch:
            ref_chosen_logps_0 = batch["ref_chosen_logps_0"]
            ref_chosen_logps_1 = batch["ref_chosen_logps_1"]
            ref_rejected_logps_0 = batch["ref_rejected_logps_0"]
            ref_rejected_logps_1 = batch["ref_rejected_logps_1"]
        else:
            ref_chosen_logps_0, ref_chosen_logps_1, ref_rejected_logps_0, ref_rejected_logps_1 = self.compute_ref_log_probs(batch)

        losses, chosen_rewards, rejected_rewards, chosen_rewards_0, chosen_rewards_1, rejected_rewards_0, rejected_rewards_1 = self.dpo_loss(
            model_output["chosen_logps_0"], model_output["chosen_logps_1"], model_output["rejected_logps_0"], model_output["rejected_logps_1"],
            ref_chosen_logps_0, ref_chosen_logps_1, ref_rejected_logps_0, ref_rejected_logps_1
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = losses + self.args.rpo_alpha * model_output["nll_loss"]  # RPO loss from V3 of the paper

        if self.use_weighting:
            losses = losses * model_output["policy_weights"]

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        metrics[f"{prefix}rewards/chosen_0"] = self.accelerator.gather_for_metrics(chosen_rewards_0).mean().item()
        metrics[f"{prefix}rewards/chosen_1"] = self.accelerator.gather_for_metrics(chosen_rewards_1).mean().item()
        metrics[f"{prefix}rewards/rejected_0"] = self.accelerator.gather_for_metrics(rejected_rewards_0).mean().item()
        metrics[f"{prefix}rewards/rejected_1"] = self.accelerator.gather_for_metrics(rejected_rewards_1).mean().item()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        metrics[f"{prefix}rewards/margins"] = (
            self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item()
        )
        metrics[f"{prefix}logps/chosen_0"] = (
            self.accelerator.gather_for_metrics(model_output["chosen_logps_0"]).detach().mean().item()
        )
        metrics[f"{prefix}logps/chosen_1"] = (
            self.accelerator.gather_for_metrics(model_output["chosen_logps_1"]).detach().mean().item()
        )
        metrics[f"{prefix}logps/rejected_0"] = (
            self.accelerator.gather_for_metrics(model_output["rejected_logps_0"]).detach().mean().item()
        )
        metrics[f"{prefix}logps/rejected_1"] = (
            self.accelerator.gather_for_metrics(model_output["rejected_logps_1"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["mean_chosen_logits"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["mean_rejected_logits"]).detach().mean().item()
        )
        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = (
                self.accelerator.gather_for_metrics(model_output["nll_loss"]).detach().mean().item()
            )
        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = (
                self.accelerator.gather_for_metrics(model_output["aux_loss"]).detach().mean().item()
            )

        return losses.mean(), metrics