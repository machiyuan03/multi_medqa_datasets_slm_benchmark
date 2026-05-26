# Multi-MedQA Datasets SLM Benchmark

本项目用于在多个医学选择题 / MedQA 数据集上评估小语言模型


| Model name              | MedQA / USMLE | MedMCQA | PubMedQA | MedXpertQA | MMLU medical subsets | MMedBench 多语种 | AfriMedQA | CRAFT-MedQA |
| ----------------------- | ------------- | ------- | -------- | ---------- | -------------------- | ------------- | --------- | ----------- |
| Qwen3.5-0.8B            | 0.4252        | 0.3705  |          |            |                      |               |           |             |
| Qwen2.5-0.5B-Instruct   | 0.3379        | 0.2245  |          |            |                      |               |           |             |
| Qwen3-0.6B              | 0.2860        | 0.2087  |          |            |                      |               |           |             |
| Granite 4.0 350M        | 0.2274        | 0.2675  |          |            |                      |               |           |             |
| Gemma 3 270M IT         | 0.2019        | 0.2338  |          |            |                      |               |           |             |
| SmolLM2-360M-Instruct   | 0.1832        | 0.2235  |          |            |                      |               |           |             |
| OLMo-2-0425-1B-Instruct | 0.1648        | 0.1554  |          |            |                      |               |           |             |
| Llama 3.2 1B Instruct   | 0.1461        | 0.0822  |          |            |                      |               |           |             |
| MiniCPM4-0.5B           | 0.1299        | 0.1181  |          |            |                      |               |           |             |
| LFM2-350M               | 0.1132        | 0.2587  |          |            |                      |               |           |             |


## Adaptation


| Model name              | MedQA / USMLE | MedMCQA              | PubMedQA | MedXpertQA | MMLU medical subsets | MMedBench 多语种 | AfriMedQA | CRAFT-MedQA |
| ----------------------- | ------------- | -------------------- | -------- | ---------- | -------------------- | ------------- | --------- | ----------- |
| Qwen3.5-0.8B            | 默认            | 默认                   |          |            |                      |               |           |             |
| Qwen2.5-0.5B-Instruct   | 默认            | 默认                   |          |            |                      |               |           |             |
| Qwen3-0.6B              | 默认            | `--no-chat-template` |          |            |                      |               |           |             |
| Granite 4.0 350M        | 默认            | 默认                   |          |            |                      |               |           |             |
| Gemma 3 270M IT         | CT 口径         | CT 口径                |          |            |                      |               |           |             |
| SmolLM2-360M-Instruct   | 默认            | 默认                   |          |            |                      |               |           |             |
| OLMo-2-0425-1B-Instruct | 默认            | `--no-chat-template` |          |            |                      |               |           |             |
| Llama 3.2 1B Instruct   | 默认            | 默认；resume `batch=1`  |          |            |                      |               |           |             |
| MiniCPM4-0.5B           | 默认            | 默认                   |          |            |                      |               |           |             |
| LFM2-350M               | 默认            | 默认                   |          |            |                      |               |           |             |


