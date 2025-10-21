# Specify Privacy Yourself: Assessing Inference-Time Personalized Privacy Preservation Ability of Large Vision-Language Model

### [Paper PDF](./assets/paper.pdf)

The official implementation of ACM Multimedia 2025 BNI Oral paper "Specify Privacy Yourself: Assessing Inference-Time Personalized Privacy Preservation Ability of Large Vision-Language Model".

<div align=center>
<img src="assets/illustration.jpg" width="100%"/>
Illustration with cases of our proposed Inference-Time Personalized Privacy Protection task.
</div>

## News
[2025-10-21] We have released the code and data for both evaluation and training!🎉 A preprint copy of our paper is also available [here](./assets/paper.pdf).

[2025-08-08] This repository has been created.

[2025-08-01] Our paper has been accepted by ACM Multimedia 2025 as a BNI Oral presentation!🎉 

## Abstract
Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities but raise significant _privacy_ concerns due to their abilities to infer sensitive personal information from images with high precision. While current LVLMs are relatively well aligned to protect universal privacy, _e.g._, credit card data, we argue that privacy is inherently personalized and context-dependent. This work pivots towards a novel task: _can LVLMs achieve Inference-Time Personalized Privacy Protection (**ITP$`^3`$**), allowing users to dynamically specify privacy boundaries through language specifications?_ To this end, we present **SPY-Bench**, the first systematic assessment of ITP$`^3`$ ability, which comprises (1) 32,700 unique samples with image-question pairs and personalized privacy instructions across 67 categories and 24 real-world scenarios, and (2) novel metrics grounded in user specifications and context awareness. Benchmarking the ITP$`^3`$ ability of 21 SOTA LVLMs, we reveal that: (i) most models, even the top-performing o4-mini, perform poorly, with only ~24% compliance accuracy; (ii) they show quite limited contextual privacy understanding capability. Therefore, we implemented initial ITP$`^3`$ alignment methods, including a novel Noise Contrastive Alignment variant which achieves 96.88% accuracy while maintaining reasonable general performance. These results mark an initial step towards the ethical deployment of more controllable LVLMs.

## Environment Setup

First, clone this repository to your local machine and navigate to the project directory:

```bash
git clone https://github.com/achernarwang/specify-privacy-yourself
cd specify-privacy-yourself
```

Then prepare the python environment with the following commands:

```bash
conda create -n spy python=3.12 -y
conda activate spy
pip install uv
uv pip install vllm qwen-vl-utils accelerate deepspeed tensorboard trl==0.15.0 liger-kernel==0.5.3
uv pip install flash-attn --no-build-isolation
```

## Evaluating with SPY-Bench

### Download Images

Download the test image dataset of [VISPR](https://tribhuvanesh.github.io/vpa/) from [this link](https://datasets.d2.mpi-inf.mpg.de/orekondy17iccv/test2017.tar.gz) and extract it to `benchmark/data/images/`. After extraction, the `benchmark` directory structure should look like this:

```
benchmark/
├── data/
│   ├── images/
│   │   └── test2017
│   │       ├── 2017_10000580.jpg
│   │       └── ...
│   ├── label2text.json
│   └── ...
└── ...
```

### Prepare Models

A complete list with download links of the evaluated LVLMs and fine-tuned checkpoints in our paper is provided in [Additional Information](#additional-information), though our implementation can theoretically benchmark any LVLM that is supported by [vLLM](https://github.com/vllm-project/vllm) or has a serving endpoint compatible with [OpenAI API](https://github.com/openai/openai-python).

If you want to evaluate an API model, please set the API key (and Endpoint URL if needed) in the `benchmark/.env`:

```bash
API_KEY = "<YOUR_API_KEY>"
BASE_URL = "<ENDPOINT_URL>" # optional if using models served by OpenAI
```

### Evaluation Steps

1. Generate the responses of the evaluated models:
   ```bash
   cd benchmark
   # For open-sourced models (with distractors)
   python generate.py --model path/to/your/model --gpu_id 0 --batch_size 64 --add_distractors --result_dir results/with_distractors 
   # For API models (without distractors)
   python generate.py --model <your_model_name_or_id> --batch_size 64 --result_dir results/without_distractors
   ```
   The argument `--add_distractors` indicates whether to include distractor privacy instructions in the evaluation. If specified, the model needs to identify the correct privacy instruction among multiple distractors. Other arguments are self-explanatory and can be checked by `python generate.py --help`.

2. **[Optional]** If you want to evaluate the self-moderation method described in section 4.1 of our paper, please run the following command after step 1:
   ```bash
   python generate_self_mod.py --file results/with_distractors/resp/<generated_file>.jsonl --model path/to/your/model --gpu_id 0 --batch_size 64 --result_dir results/with_distractors_self_mod
   ```
   The evaluated model in this step should be the same as that in step 1.

3. Evaluate the generated results with a judgment model (specified by `--eval_model`). The judgment model does not require multimodal capability, so you can use pure LLMs in this step. If you decide to use an API model (in our paper we use GPT-4o), please also set `EVAL_API_KEY` and `EVAL_BASE_URL` in the `benchmark/.env` file.
   ```bash
   # using open-sourced models (require vLLM support) as judgment model
   python evaluate.py --eval_model /path/to/your/model --gpu_id 0 --batch_size 64 --result_dir results/with_distractors  --result_file resp/<generated_file>.jsonl
   # using API models as judgment model
   python evaluate.py --eval_model <your_model_name_or_id> --batch_size 64 --result_dir results/with_distractors  --result_file resp/<generated_file>.jsonl
   ```
   If you are evaluating the results generated with self-moderation method, please also specify the argument `--resp_k` as `resp_3` in the command above.

4. Calculate the metrics for the evaluation results:
   ```bash
   python metrics.py -f eval/<evaluated_file>.jsonl --result_dir results/with_distractors
   ```
   If you are evaluating the results generated with self-moderation method, please specify the argument `--eval_k` as `eval_3` in the command above. To calculate the overall score across SPY-Bench and general benchmarks ([MMMU](https://arxiv.org/abs/2311.16502), [OCRBench](http://arxiv.org/abs/2305.07895), [MME](https://arxiv.org/abs/2306.13394)), you can use the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to obtain the results of these general benchmark and then specify the `-g` argument with the path to the general benchmark results file when running `metrics.py`.

## Training

### Prepare the training data (SPY-Tune)

First, download the training image dataset of [VISPR](https://tribhuvanesh.github.io/vpa/) from [this link](https://datasets.d2.mpi-inf.mpg.de/orekondy17iccv/train2017.tar.gz) and extract it to `train/data/images/`. Then download the training annotations from [this link](https://drive.google.com/file/d/1FvLLls9g-VlA3hVMBIxc2TmcpA1bWvZW/view?usp=sharing) and move it to `train/data/`.

The `train` directory should now look like this:

```
train/
├── configs/
├── data/
│   ├── images/
│   │   └── train2017
│   │       ├── 2017_10001018.jpg
│   │       └── ...
│   ├── train_data.json
│   ├── eval_data.json
│   └── ...
└── ...
```

### Training Scripts

We provide training scripts for all the methods adopted in our paper, including SFT (`train/train_sft.py`), [DPO](https://arxiv.org/abs/2305.18290) / [NCA](https://arxiv.org/abs/2402.05369) (`train/train_dpo.py`), and NCA-P (`train/train_our.py`). Below is an example command for training with NCA-P:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
accelerate launch --config_file configs/deepspeed_zero2.yaml --num_processes 8 \ # Number of using GPUs
   train_our.py \
   --model_name_or_path /path/to/Qwen2-VL-7B-Instruct \
   --train_data_path data/train_data.json \
   --eval_data_path data/eval_data.json \
   --label_path data/label2text.json \
   --image_folder data \
   --shuffle True \
   --add_distractors True \
   --min_pixels 200704 \
   --max_pixels 1003520 \
   --num_train_epochs 3.0 \
   --save_strategy "epoch" \
   --logging_steps 10 \
   --eval_steps 100 \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 1 \
   --gradient_checkpointing \
   --learning_rate 3e-6 \
   --loss_type "nca_priv" \
   --beta 0.01 \
   --weight_decay 0.05 \
   --warmup_ratio 0.1 \
   --lr_scheduler_type "cosine" \
   --bf16 \
   --tf32 True \
   --torch_dtype bfloat16 \
   --use_liger \
   --attn_implementation flash_attention_2 \
   --output_dir runs/q2_ncap_b32_l3e-6_b001_e3_wd005_wr01 \
   --save_only_model True \
   --report_to tensorboard
```
You could refer to the `train/scripts/` for more example commands.

> [!Tip]
> If you are encountering CUDA out-of-memory issues during training, other than reducing the training batch size, you could also try to adjust the deepspeed settings in your Accelerate configuration file (under `train/configs/`) following the instructions [here](https://huggingface.co/docs/transformers/v4.49.0/en/deepspeed#select-a-zero-stage).

## Acknowledgements
We gratefully acknowledge the developers and contributors of [VISPR](https://arxiv.org/abs/1703.10660), [🤗Huggingface Libraries](https://huggingface.co/docs), [vLLM Project](https://github.com/vllm-project/vllm), and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), upon which our work is built. We also extend our gratitude to the authors of all evaluated models (see the list below) for sharing the model weights or service endpoints with the research community.

## Additional Information

In our paper, we evaluated the following LVLMs with SPY-Bench:

| Model Name | Model Type | Source |
|------------|------------|--------|
| LLaVA 1.5 13B | Open-source | [🤗 HuggingFace](https://huggingface.co/liuhaotian/llava-v1.5-13b) |
| LLaVA NeXT Vicuna 13B | Open-source | [🤗 HuggingFace](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf) |
| LLaVA OneVision Qwen2 7B | Open-source | [🤗 HuggingFace](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov) |
| Llama 3.2 11B Vision Instruct | Open-source | [🤗 HuggingFace](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) |
| Pixtral 12B | Open-source | [🤗 HuggingFace](https://huggingface.co/mistralai/Pixtral-12B-2409) |
| GLM 4V 9B | Open-source | [🤗 HuggingFace](https://huggingface.co/THUDM/glm-4v-9b) |
| Deepseek VL2 | Open-source | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/deepseek-vl2) |
| InternVL 2.5 4B/8B/38B/78B | Open-source | [🤗 HuggingFace](https://huggingface.co/collections/OpenGVLab/internvl25-673e1019b66e2218f68d7c1c) |
| Qwen2 VL 7B Instruct | Open-source | [🤗 HuggingFace](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) |
| Qwen2.5 VL 3B/7B/32B/72B Instruct | Open-source | [🤗 HuggingFace](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5) |
| Phi 4 Multimodal Instruct | Open-source | [🤗 HuggingFace](https://huggingface.co/microsoft/Phi-4) |
| Mistral Small 3.1 24B Instruct 2503 | Open-source | [🤗 HuggingFace](https://huggingface.co/mistralai/Mistral-Small-Instruct-2503) |
| GPT 4o 2024-11-20 | Proprietary | [OpenAI](https://platform.openai.com/docs/models/gpt-4o) |
| Gemini 2.0 Flash | Proprietary | [Google AI](https://ai.google.dev/gemini-api/docs/models#gemini-2.0-flash) |
| o4-mini 2025-04-16 | Proprietary | [OpenAI](https://platform.openai.com/docs/models/o4-mini) |

The information and download link of the fine-tuned checkpoints used in our paper are provided below:

| Base Model | Finetuning Method | Checkpoint Link |
|-------|----------------|-----------------|
| Qwen2-VL-7B-Instruct | SFT | [🤗 HuggingFace](https://huggingface.co/achernarwang/SPY_Qwen2-VL-7B-Instruct_SFT) |
| Qwen2-VL-7B-Instruct | DPO | [🤗 HuggingFace](https://huggingface.co/achernarwang/SPY_Qwen2-VL-7B-Instruct_DPO) |
| Qwen2-VL-7B-Instruct | NCA | [🤗 HuggingFace](https://huggingface.co/achernarwang/SPY_Qwen2-VL-7B-Instruct_NCA) |
| Qwen2-VL-7B-Instruct | NCA-P | [🤗 HuggingFace](https://huggingface.co/achernarwang/SPY_Qwen2-VL-7B-Instruct_NCA-P) |
