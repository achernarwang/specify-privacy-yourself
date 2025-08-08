# Specify Privacy Yourself: Assessing Inference-Time Personalized Privacy Preservation Ability of Large Vision-Language Model

The official implementation of ACM Multimedia 2025 BNI Oral paper "Specify Privacy Yourself: Assessing Inference-Time Personalized Privacy Preservation Ability of Large Vision-Language Model"

> [!note]
> The code and dataset will be publicly available before the opening of ACM Multimedia 2025 (October 27, 2025)



## News
[2025-08-08] Repository created!

## Abstract
Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities but raise significant _privacy_ concerns due to their abilities to infer sensitive personal information from images with high precision. While current LVLMs are relatively well aligned to protect universal privacy, _e.g._, credit card data, we argue that privacy is inherently personalized and context-dependent. This work pivots towards a novel task: _can LVLMs achieve Inference-Time Personalized Privacy Protection (**ITP$`^3`$**), allowing users to dynamically specify privacy boundaries through language specifications?_ To this end, we present **SPY-Bench**, the first systematic assessment of ITP$`^3`$ ability, which comprises (1) 32,700 unique samples with image-question pairs and personalized privacy instructions across 67 categories and 24 real-world scenarios, and (2) novel metrics grounded in user specifications and context awareness. Benchmarking the ITP$`^3`$ ability of 21 SOTA LVLMs, we reveal that: (i) most models, even the top-performing o4-mini, perform poorly, with only ~24% compliance accuracy; (ii) they show quite limited contextual privacy understanding capability. Therefore, we implemented initial ITP$`^3`$ alignment methods, including a novel Noise Contrastive Alignment variant which achieves 96.88% accuracy while maintaining reasonable general performance. These results mark an initial step towards the ethical deployment of more controllable LVLMs.
