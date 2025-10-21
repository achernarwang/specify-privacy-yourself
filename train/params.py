from typing import Optional
from dataclasses import dataclass, field

from trl import DPOConfig

@dataclass
class ScriptArguments:
    train_data_path: str = field(
        default=None,
        metadata={"help": "Path to the train data file."},
    )
    eval_data_path: str = field(
        default=None,
        metadata={"help": "Path to the eval data file."},
    )
    label_path: str = field(
        default=None,
        metadata={"help": "Path to the data file."},
    )
    image_folder: str = field(
        default=None,
        metadata={"help": "Path to the image folder."},
    )
    add_distractors: bool = field(
        default=True,
        metadata={"help": "Add other labels to the dataset."},
    )
    num_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Number of training samples."},
    )
    min_pixels: Optional[int] = field(
        default=None, # default for Qwen VL 3136
        metadata={"help": "Minimum number of pixels."},
    )
    max_pixels: Optional[int] = field(
        default=None, # default for Qwen VL 12845056
        metadata={"help": "Maximum number of pixels."},
    )
    eval_temperature: Optional[float] = field(
        default=0.0,
        metadata={"help": "Temperature for evaluation."},
    )
    eval_max_new_tokens: Optional[int] = field(
        default=128,
        metadata={"help": "Maximum number of new tokens for evaluation."},
    )
    num_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Number of prompts for evaluation."},
    )
    project: str = field(
        default=None,
        metadata={"help": "Wandb project name."},
    )
    train_name: str = field(
        default=None,
        metadata={"help": "Wandb run name."},
    )
    wandb_mode: str = field(
        default="disabled",
        metadata={"help": "Wandb mode."},
    )
    ignore_bias_buffers: bool = field(
        default=False,
        metadata={
            "help": "Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid "
            "scalar type, inplace operation. See "
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992."
        },
    )
    shuffle: bool = field(
        default=False,
        metadata={"help": "Shuffle the dataset."},
    )

@dataclass
class CustomDPOConfig(DPOConfig):
    use_liger: bool = field(
        default=False,
        metadata={"help": "Monkey patch the model with Liger kernels to increase throughput and reduce memory usage."},
    )
    max_prompt_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum length of the prompt."},
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum length of the full sequence (prompt + completion)."},
    )
    loss_type: str = field(
        default="sigmoid",
        metadata={
            "help": "Type of loss to use.",
            "choices": [
                "sigmoid",
                "hinge",
                "ipo",
                "exo_pair",
                "nca_pair",
                "nca_priv",
                "robust",
                "bco_pair",
                "sppo_hard",
                "aot",
                "aot_pair",
                "discopop",
                "apo_zero",
                "apo_down",
            ],
        },
    )