

# Especifica tu propia privacidad: Evaluando la capacidad de preservación de privacidad personalizada en tiempo de inferencia de los grandes modelos de visión y lenguaje

### [PDF del Artículo](./assets/paper.pdf) | [ACM DL](https://dl.acm.org/doi/10.1145/3746027.3758156)

Implementación oficial del artículo oral BNI de ACM Multimedia 2025 "Especifica tu propia privacidad: Evaluando la capacidad de preservación de privacidad personalizada en tiempo de inferencia de los grandes modelos de visión y lenguaje".

<div align=center>
<img src="assets/illustration.jpg" width="100%"/>
Ilustración con casos de nuestra tarea propuesta de Protección de Privacidad Personalizada en Tiempo de Inferencia.
</div>

## Noticias
[2025-10-21] 🎉 ¡Hemos publicado el código y los datos tanto para la evaluación como para el entrenamiento! También está disponible una versión preliminar de nuestro artículo [aquí](./assets/paper.pdf).

[2025-08-08] 🚀 Este repositorio ha sido creado.

[2025-08-01] 🎉 ¡Nuestro artículo ha sido aceptado por ACM Multimedia 2025 como presentación oral BNI! 

## Resumen
Los grandes modelos de visión y lenguaje (LVLMs) han demostrado capacidades notables, pero generan importantes preocupaciones de _privacidad_ debido a su habilidad para inferir información personal sensible a partir de imágenes con alta precisión. Si bien los LVLMs actuales están relativamente bien alineados para proteger la privacidad universal, _por ejemplo_, datos de tarjetas de crédito, argumentamos que la privacidad es inherentemente personalizada y dependiente del contexto. Este trabajo se enfoca en una tarea novedosa: _¿pueden los LVLMs lograr la Protección de Privacidad Personalizada en Tiempo de Inferencia (**ITP$`^3`$**), permitiendo a los usuarios especificar dinámicamente los límites de privacidad mediante instrucciones en lenguaje natural?_ Para ello, presentamos **SPY-Bench**, la primera evaluación sistemática de la capacidad ITP$`^3`$, que comprende (1) 32.700 muestras únicas con pares de imagen-pregunta e instrucciones de privacidad personalizada en 67 categorías y 24 escenarios del mundo real, y (2) métricas novedosas basadas en especificaciones de usuario y conciencia contextual. Al evaluar la capacidad ITP$`^3`$ de 21 LVLMs estado del arte, revelamos que: (i) la mayoría de los modelos, incluso el de mejor rendimiento o4-mini, rinden pobremente, con solo ~24% de precisión de cumplimiento; (ii) muestran una capacidad de comprensión contextual de la privacidad bastante limitada. Por lo tanto, implementamos métodos iniciales de alineación ITP$`^3`$, incluida una variante novedosa de Alineación Contrastiva de Ruido que alcanza un 96.88% de precisión manteniendo un rendimiento general razonable. Estos resultados marcan un paso inicial hacia el despliegue ético de LVLMs más controlables.

## Configuración del Entorno

Primero, clona este repositorio en tu máquina local y navega al directorio del proyecto:

```bash
git clone https://github.com/achernarwang/specify-privacy-yourself
cd specify-privacy-yourself
```

Luego, prepara el entorno de Python con los siguientes comandos:

```bash
conda create -n spy python=3.12 -y
conda activate spy
pip install uv
uv pip install vllm qwen-vl-utils accelerate deepspeed tensorboard trl==0.15.0 liger-kernel==0.5.3
uv pip install flash-attn --no-build-isolation
```

## Evaluación con SPY-Bench

### Descargar Imágenes

Descarga el conjunto de datos de imágenes de prueba de [VISPR](https://tribhuvanesh.github.io/vpa/) desde [este enlace](https://datasets.d2.mpi-inf.mpg.de/orekondy17iccv/test2017.tar.gz) y extraelo en `benchmark/data/images/`. Después de la extracción, la estructura del directorio `benchmark` debería verse así:

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

### Preparar Modelos

Una lista completa con enlaces de descarga de los LVLMs evaluados y los puntos de control ajustados en nuestro artículo se proporciona en [Información Adicional](#información-adicional), aunque nuestra implementación puede evaluar teóricamente cualquier LVLM compatible con [vLLM](https://github.com/vllm-project/vllm) o que tenga un punto de conexión compatible con la [API de OpenAI](https://github.com/openai/openai-python).

Si deseas evaluar un modelo basado en API, configura la clave de API (y la URL del punto de conexión si es necesario) en `benchmark/.env`:

```bash
API_KEY = "<YOUR_API_KEY>"
BASE_URL = "<ENDPOINT_URL>" # opcional si se usan modelos servidos por OpenAI
```

### Pasos de Evaluación

1. Genera las respuestas de los modelos evaluados:
   ```bash
   cd benchmark
   # Para modelos de código abierto (con distractores)
   python generate.py --model path/to/your/model --gpu_id 0 --batch_size 64 --add_distractors --result_dir results/with_distractors 
   # Para modelos de API (sin distractores)
   python generate.py --model <your_model_name_or_id> --batch_size 64 --result_dir results/without_distractors
   ```
   El argumento `--add_distractors` indica si se deben incluir instrucciones de privacidad distractoras en la evaluación. Si se especifica, el modelo debe identificar la instrucción de privacidad correcta entre varios distractores. Los demás argumentos son autoexplicativos y se pueden consultar con `python generate.py --help`.

2. **[Opcional]** Si deseas evaluar el método de automoderación descrito en la sección 4.1 de nuestro artículo, ejecuta el siguiente comando después del paso 1:
   ```bash
   python generate_self_mod.py --file results/with_distractors/resp/<generated_file>.jsonl --model path/to/your/model --gpu_id 0 --batch_size 64 --result_dir results/with_distractors_self_mod
   ```
   El modelo evaluado en este paso debe ser el mismo que en el paso 1.

3. Evalúa los resultados generados con un modelo de juicio (especificado por `--eval_model`). El modelo de juicio no requiere capacidad multimodal, por lo que puedes usar LLMs puros en este paso. Si decides usar un modelo de API (en nuestro artículo usamos GPT-4o), también configura `EVAL_API_KEY` y `EVAL_BASE_URL` en el archivo `benchmark/.env`.
   ```bash
   # usando modelos de código abierto (requieren soporte de vLLM) como modelo de juicio
   python evaluate.py --eval_model /path/to/your/model --gpu_id 0 --batch_size 64 --result_dir results/with_distractors  --result_file resp/<generated_file>.jsonl
   # usando modelos de API como modelo de juicio
   python evaluate.py --eval_model <your_model_name_or_id> --batch_size 64 --result_dir results/with_distractors  --result_file resp/<generated_file>.jsonl
   ```
   Si estás evaluando los resultados generados con el método de automoderación, también especifica el argumento `--resp_k` como `resp_3` en el comando anterior.

4. Calcula las métricas para los resultados de la evaluación:
   ```bash
   python metrics.py -f eval/<evaluated_file>.jsonl --result_dir results/with_distractors
   ```
   Si estás evaluando los resultados generados con el método de automoderación, especifica el argumento `--eval_k` como `eval_3` en el comando anterior. Para calcular la puntuación general en SPY-Bench y conjuntos de benchmarks generales ([MMMU](https://arxiv.org/abs/2311.16502), [OCRBench](http://arxiv.org/abs/2305.07895), [MME](https://arxiv.org/abs/2306.13394)), puedes usar [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) para obtener los resultados de estos benchmarks generales y luego especificar el argumento `-g` con la ruta al archivo de resultados de los benchmarks generales al ejecutar `metrics.py`.

## Entrenamiento

### Preparar los datos de entrenamiento (SPY-Tune)

Primero, descarga el conjunto de datos de imágenes de entrenamiento de [VISPR](https://tribhuvanesh.github.io/vpa/) desde [este enlace](https://datasets.d2.mpi-inf.mpg.de/orekondy17iccv/train2017.tar.gz) y extraelo en `train/data/images/`. Luego, descarga las anotaciones de entrenamiento desde [este enlace](https://drive.google.com/file/d/1FvLLls9g-VlA3hVMBIxc2TmcpA1bWvZW/view?usp=sharing) y muévelas a `train/data/`.

El directorio `train` ahora debería verse así:

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

### Scripts de Entrenamiento

Proporcionamos scripts de entrenamiento para todos los métodos adoptados en nuestro artículo, incluyendo SFT (`train/train_sft.py`), [DPO](https://arxiv.org/abs/2305.18290) / [NCA](https://arxiv.org/abs/2402.05369) (`train/train_dpo.py`), y NCA-P (`train/train_our.py`). A continuación se muestra un ejemplo de comando para entrenar con NCA-P:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
accelerate launch --config_file configs/deepspeed_zero2.yaml --num_processes 8 \ # Número de GPUs a utilizar
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
Puedes consultar `train/scripts/` para más ejemplos de comandos.

> [!Tip]
> Si experimentas problemas de memoria CUDA agotada durante el entrenamiento, además de reducir el tamaño del lote de entrenamiento, también puedes intentar ajustar la configuración de deepspeed en tu archivo de configuración de Accelerate (en `train/configs/`) siguiendo las instrucciones [aquí](https://huggingface.co/docs/transformers/v4.49.0/en/deepspeed#select-a-zero-stage).

## Agradecimientos
Agradecemos profundamente a los desarrolladores y colaboradores de [VISPR](https://arxiv.org/abs/1703.10660), [🤗Huggingface Libraries](https://huggingface.co/docs), [vLLM Project](https://github.com/vllm-project/vllm), y [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), sobre los cuales se basa nuestro trabajo. También extendemos nuestro agradecimiento a los autores de todos los modelos evaluados (ver lista a continuación) por compartir los pesos de los modelos o puntos de conexión con la comunidad de investigación.

## Citación

Si consideras que este repositorio es útil para tu investigación, por favor considera citar nuestro trabajo:

```
@inproceedings{wang2025specify,
  title={Specify Privacy Yourself: Assessing Inference-Time Personalized Privacy Preservation Ability of Large Vision-Language Models},
  author={Wang, Xingqi and Yi, Xiaoyuan and Xie, Xing and Jia, Jia},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={12304--12313},
  year={2025}
}
```

## Información Adicional

En nuestro artículo, evaluamos los siguientes LVLMs con SPY-Bench:

| Nombre del Modelo | Tipo de Modelo | Fuente |
|------------|------------|--------|
| LLaVA 1.5 13B | Código abierto | [🤗 HuggingFace](https://huggingface.co/llava-hf/llava-1.5-13b-hf) |
| LLaVA NeXT Vicuna 13B | Código abierto | [🤗 HuggingFace](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf) |
| LLaVA OneVision Qwen2 7B | Código abierto | [🤗 HuggingFace](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf) |
| Llama 3.2 11B Vision Instruct | Código abierto | [🤗 HuggingFace](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) |
| Pixtral 12B | Código abierto | [🤗 HuggingFace](https://huggingface.co/mistralai/Pixtral-12B-2409) |
| GLM 4V 9B | Código abierto | [🤗 HuggingFace](https://huggingface.co/THUDM/glm-4v-9b) |
| Deepseek VL2 | Código abierto | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/deepseek-vl2) |
| InternVL 2.5 4B/8B/38B/78B | Código abierto | [🤗 HuggingFace](https://huggingface.co/collections/OpenGVLab/internvl25-673e1019b66e2218f68d7c1c) |
| Qwen2 VL 7B Instruct | Código abierto | [🤗 HuggingFace](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) |
| Qwen2.5 VL 3B/7B/32B/72B Instruct | Código abierto | [🤗 HuggingFace](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5) |
| Phi 4 Multimodal Instruct | Código abierto | [🤗 HuggingFace](https://huggingface.co/microsoft/Phi-4) |
| Mistral Small 3.1 24B Instruct 2503 | Código abierto | [🤗 HuggingFace](https://huggingface.co/mistralai/Mistral-Small-Instruct-2503) |
| GPT 4o 2024-11-20 | Propietario | [OpenAI](https://platform.openai.com/docs/models/gpt-4o) |
| Gemini 2.0 Flash | Propietario | [Google AI](https://ai.google.dev/gemini-api/docs/models#gemini-2.0-flash) |
| o4-mini 2025-04-16 | Propietario | [OpenAI](https://platform.openai.com/docs/models/o4-mini) |

La información y el enlace de descarga de los puntos de control ajustados utilizados en nuestro artículo se proporcionan a continuación:

| Modelo Base | Método de Ajuste Fino | Enlace del Punto de Control |
|-------|----------------|-----------------|
| Qwen2-VL-7B-Instruct | SFT | [🤗 HuggingFace](https://huggingface.co/achernarwang/SPY_Qwen2-VL-7B-Instruct_SFT) |
| Qwen2-VL-7B-Instruct | DPO | [🤗 HuggingFace](https://huggingface.co/achernarwang/SPY_Qwen2-VL-7B-Instruct_DPO) |
| Qwen2-VL-7B-Instruct | NCA | [🤗 HuggingFace](https://huggingface.co/achernarwang/SPY_Qwen2-VL-7B-Instruct_NCA) |
| Qwen2-VL-7B-Instruct | NCA-P | [🤗 HuggingFace](https://huggingface.co/achernarwang/SPY_Qwen2-VL-7B-Instruct_NCA-P) |
