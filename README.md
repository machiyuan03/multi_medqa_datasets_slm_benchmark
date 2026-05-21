# Small Language Model Medical QA Benchmark

This repository contains a lightweight evaluation pipeline for benchmarking small language models on MedQA-style multiple-choice medical questions in English, Simplified Chinese, and Traditional Chinese.

The public repository includes the inference/evaluation code and model adapter configuration. Dataset files, generated outputs, and internal run logs are intentionally excluded from version control.

## Evaluation Protocol

- Test set: canonical MedQA test files for Mainland Chinese, Taiwan Chinese, and US English, totaling 6,112 questions.
- Decoding: greedy generation with `max_new_tokens=64`.
- Precision: full precision (`torch.float32`) for all reported runs.
- Prompt: fixed benchmark prompt across models; model-specific chat templates are used only as tokenizer/model interface adapters when required.
- Evaluation: answer labels are extracted from raw generated text with a shared parser, and invalid predictions are counted as incorrect.

## Results

| Model | Accuracy | Confidence | Notes |
| --- | ---: | --- | --- |
| Qwen3.5-0.8B | 0.4252 | High | Full 6,112-example run with no invalid predictions; answers were consistently extractable despite frequent generation-limit stops. |
| Qwen2.5-0.5B-Instruct | 0.3379 | Medium-high | Full run completed with 317 invalid predictions; a high truncation rate was observed, but most final answers remained parseable. |
| Qwen3-0.6B | 0.2860 | High | Outputs were generally stable under the shared parser, with 1,405 invalid predictions and moderate truncation. |
| Granite 4.0 350M | 0.2274 | High | Strongest format compliance among the evaluated sub-billion models, with 103 invalid predictions and minimal truncation. |
| Gemma 3 270M IT | 0.2019 | Medium | Full run with 318 invalid predictions; short-generation sensitivity suggests some residual effect from the 64-token cap. |
| SmolLM2-360M-Instruct | 0.1832 | Medium | Full run completed, but many answers required fallback parsing and 879 predictions were invalid. |
| OLMo-2-0425-1B-Instruct | 0.1648 | Medium | Substantial invalid-output rate was observed; token-limit diagnostics did not indicate a large systematic accuracy gain from longer generation. |
| Llama 3.2 1B Instruct | 0.1461 | Medium | Results were dominated by formatting errors and answer-distribution bias rather than an obvious generation-length artifact. |
| MiniCPM4-0.5B | 0.1299 | Medium | High invalid-output rate; limited longer-generation diagnostics showed only minor improvement. |
| LFM2-350M | 0.1132 | Medium-low | High invalid-output rate and fallback parsing make this result less robust than the higher-ranked models. |

Notes:

- The reported Qwen3.5-0.8B score uses the full-precision run. An earlier `bfloat16` run is excluded from the public table.
- The reported Gemma 3 270M IT score uses the chat-template adapted run. The raw-prompt run is excluded from the public table.

## Usage

Run inference with a local Hugging Face model:

```bash
python infer.py --model-path /path/to/model --model-name model-name
```

Evaluate a generated JSONL file:

```bash
python eva.py outputs/model-name/model-name.jsonl --summary-only
```

Model-specific adapter settings are defined in `configs/models.json`.

## Citation

This benchmark uses the MedQA dataset:

```bibtex
@article{jin2021disease,
  title={What disease does this patient have? a large-scale open domain question answering dataset from medical exams},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
  journal={Applied Sciences},
  volume={11},
  number={14},
  pages={6421},
  year={2021},
  publisher={MDPI}
}
```
