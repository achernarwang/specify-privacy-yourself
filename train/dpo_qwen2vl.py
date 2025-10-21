import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Union
from transformers.data.data_collator import DataCollatorMixin
from trl.data_utils import maybe_apply_chat_template
from trl.trainer.utils import pad, pad_to_length, selective_log_softmax, flush_left

class Qwen2VLDataCollatorForPreference(DataCollatorMixin):
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
        examples = [{**maybe_apply_chat_template(example, self.processor), "images": example["images"]} for example in examples]

        # Tokenizing
        processed_examples = []
        for features in examples:
            processed_features = self.processor(images=features["images"], text=features["prompt"], add_special_tokens=False)
            prompt_input_ids = processed_features["input_ids"][0]
            pixel_values = processed_features["pixel_values"]
            chosen_input_ids = self.tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]
            rejected_input_ids = self.tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]

            chosen_input_ids = chosen_input_ids + [self.tokenizer.eos_token_id]
            rejected_input_ids = rejected_input_ids + [self.tokenizer.eos_token_id]

            # Truncate prompt and completion sequences
            if self.args.max_prompt_length is not None:
                prompt_input_ids = prompt_input_ids[-self.args.max_prompt_length:]
            if self.args.max_completion_length is not None:
                chosen_input_ids = chosen_input_ids[:self.args.max_completion_length]
                rejected_input_ids = rejected_input_ids[:self.args.max_completion_length]

            output = {
                "prompt_input_ids": prompt_input_ids,
                "pixel_values": pixel_values,
                "chosen_input_ids": chosen_input_ids,
                "rejected_input_ids": rejected_input_ids,
            }

            if "pixel_attention_mask" in processed_features:
                output["pixel_attention_mask"] = processed_features["pixel_attention_mask"][0]
            if "image_sizes" in processed_features:
                output["image_sizes"] = processed_features["image_sizes"][0]
            if "image_grid_thw" in processed_features:
                output["image_grid_thw"] = processed_features["image_grid_thw"][0]
            
            processed_examples.append(output)

        examples = processed_examples
        # Convert to tensor
        prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
        prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]
        chosen_input_ids = [torch.tensor(example["chosen_input_ids"]) for example in examples]
        chosen_attention_mask = [torch.ones_like(input_ids) for input_ids in chosen_input_ids]
        rejected_input_ids = [torch.tensor(example["rejected_input_ids"]) for example in examples]
        rejected_attention_mask = [torch.ones_like(input_ids) for input_ids in rejected_input_ids]
        if "pixel_values" in examples[0]:
            pixel_values = [torch.tensor(example["pixel_values"]) for example in examples]
        if "image_grid_thw" in examples[0]:
            image_grid_thw = [torch.tensor(example["image_grid_thw"]) for example in examples]
        if "pixel_attention_mask" in examples[0]:
            pixel_attention_mask = [torch.tensor(example["pixel_attention_mask"]) for example in examples]
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            ref_chosen_logps = torch.tensor([example["ref_chosen_logps"] for example in examples])
            ref_rejected_logps = torch.tensor([example["ref_rejected_logps"] for example in examples])

        # Pad
        output = {}
        output["prompt_input_ids"] = pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left")
        output["prompt_attention_mask"] = pad(prompt_attention_mask, padding_value=0, padding_side="left")
        output["chosen_input_ids"] = pad(chosen_input_ids, padding_value=self.pad_token_id)
        output["chosen_attention_mask"] = pad(chosen_attention_mask, padding_value=0)
        output["rejected_input_ids"] = pad(rejected_input_ids, padding_value=self.pad_token_id)
        output["rejected_attention_mask"] = pad(rejected_attention_mask, padding_value=0)
        if "pixel_values" in examples[0]:
            # output["pixel_values"] = pad(pixel_values, padding_value=0.0)
            output["pixel_values"] = torch.cat(pixel_values, dim=0)
        if "pixel_attention_mask" in examples[0]:
            output["pixel_attention_mask"] = pad(pixel_attention_mask, padding_value=0)
        if "image_sizes" in examples[0]:
            output["image_sizes"] = torch.tensor([example["image_sizes"] for example in examples])
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            output["ref_chosen_logps"] = ref_chosen_logps
            output["ref_rejected_logps"] = ref_rejected_logps
        if "image_grid_thw" in examples[0]:
            output["image_grid_thw"] = torch.stack(image_grid_thw, dim=0)

        return output

def qwen2vl_prepare_dataset(self, dataset, processing_class, args, dataset_name):

    return dataset

@staticmethod
def qwen2vl_concatenated_inputs(
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

    # For the prompt, the input_ids are the same for both the chosen and rejected responses
    output["prompt_input_ids"] = torch.cat([batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0)
    output["prompt_attention_mask"] = torch.cat(
        [batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0
    )
    if "pixel_values" in batch:
        output["pixel_values"] = torch.cat([batch["pixel_values"], batch["pixel_values"]], dim=0)

    if "pixel_attention_mask" in batch:
        output["pixel_attention_mask"] = torch.cat(
            [batch["pixel_attention_mask"], batch["pixel_attention_mask"]], dim=0
        )
    if "image_sizes" in batch:
        output["image_sizes"] = torch.cat([batch["image_sizes"], batch["image_sizes"]], dim=0)
    if "image_grid_thw" in batch:
        output["image_grid_thw"] = torch.cat([batch["image_grid_thw"], batch["image_grid_thw"]], dim=0)

    # Concatenate the chosen and rejected completions
    max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
    output["completion_input_ids"] = torch.cat(
        (
            pad_to_length(batch["chosen_input_ids"], max_completion_length, pad_value=padding_value),
            pad_to_length(batch["rejected_input_ids"], max_completion_length, pad_value=padding_value),
        ),
    )
    output["completion_attention_mask"] = torch.cat(
        (
            pad_to_length(batch["chosen_attention_mask"], max_completion_length, pad_value=0),
            pad_to_length(batch["rejected_attention_mask"], max_completion_length, pad_value=0),
        ),
    )

    return output

def qwen2vl_concatenated_forward(self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]]):
    """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

    We do this to avoid doing two forward passes, because it's faster for FSDP.
    """
    num_examples = batch["prompt_input_ids"].shape[0]

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

    prompt_input_ids = concatenated_batch["prompt_input_ids"]
    prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
    completion_input_ids = concatenated_batch["completion_input_ids"]
    completion_attention_mask = concatenated_batch["completion_attention_mask"]
    if self.is_encoder_decoder:
        labels = completion_input_ids
        labels[completion_attention_mask == 0] = self.label_pad_token_id
        outputs = model(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            labels=labels,  # we need the labels for the logits to be returned
            **model_kwargs,
        )
        logits = outputs.logits
        loss_mask = completion_attention_mask.bool()
    else:
        # Concatenate the prompt and completion inputs
        input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
        # Mask the prompt but not the completion for the loss
        loss_mask = torch.cat(
            (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
            dim=1,
        )

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
            chosen_weights = all_weights[:num_examples]
            rejected_weights = all_weights[num_examples:]
            output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)

    if self.args.rpo_alpha is not None:
        # Only use the chosen logits for the RPO loss
        chosen_logits = logits[:num_examples]
        chosen_labels = labels[:num_examples]

        # Compute the log probabilities of the labels
        output["nll_loss"] = F.cross_entropy(
            torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
        )

    if self.loss_type == "ipo":
        all_logps = all_logps / loss_mask.sum(-1)

    output["chosen_logps"] = all_logps[:num_examples]
    output["rejected_logps"] = all_logps[num_examples:]

    # Compute the mean logits
    if self.padding_free:
        # position_ids contains a sequence of range identifiers (e.g., [[0, 1, 2, 0, 1, 2, 3, ...]]).
        # There are 2*num_examples ranges in total: the first half corresponds to the chosen tokens,
        # and the second half to the rejected tokens.
        # To find the start of the rejected tokens, we look for the num_examples+1-th zero in pos_id.
        split_idx = (position_ids == 0).nonzero(as_tuple=True)[1][num_examples]
        mean_chosen_logits = logits[0, :split_idx][loss_mask[0, :split_idx]].mean()
        mean_rejected_logits = logits[0, split_idx:][loss_mask[0, split_idx:]].mean()
    else:
        mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
        mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()

    output["mean_chosen_logits"] = mean_chosen_logits
    output["mean_rejected_logits"] = mean_rejected_logits

    if self.aux_loss_enabled:
        output["aux_loss"] = outputs.aux_loss

    return output