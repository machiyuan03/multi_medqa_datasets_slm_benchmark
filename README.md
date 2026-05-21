# Benchmarking Small Language Models on Multilingual Medical Multiple-Choice QA

This repository provides a lightweight evaluation pipeline and released model outputs for benchmarking small language models on MedQA-style multiple-choice medical questions in English, Simplified Chinese, and Traditional Chinese.

The repository includes inference code, evaluation code, model adapter settings, machine-readable result summaries, and selected full-run outputs. The original dataset files are not redistributed here; users should obtain MedQA from its official source and respect the dataset license.

## Benchmark Protocol

- Dataset split: canonical MedQA test files for Mainland Chinese, Taiwan Chinese, and US English.
- Evaluation size: 6,112 questions.
- Option order: shuffled with a fixed seed (`123`) and answer labels remapped accordingly.
- Decoding: greedy generation with `max_new_tokens=64`.
- Precision: full precision (`torch.float32`) for all reported runs.
- Prompting: a fixed benchmark prompt is used across models; model-specific chat templates are treated as tokenizer/model interface adapters, not task-specific prompt changes.
- Scoring: answer labels are extracted from raw generated text with a shared parser; invalid predictions are counted as incorrect.
- Fine-tuning: no benchmark-specific fine-tuning is performed in this repository.

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

Machine-readable results are available in [`results.csv`](results.csv). Full released JSONL outputs are stored under [`Outputs/`](Outputs/).

Notes:

- The reported Qwen3.5-0.8B score uses the full-precision run in `Outputs/Qwen3.5-0.8B-fp32/`. An earlier `bfloat16` run is excluded.
- The reported Gemma 3 270M IT score uses the chat-template adapted run in `Outputs/gemma-3-270m-it-ct/`. The raw-prompt run is excluded.

## Reproducibility

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Evaluate a released output file:

```bash
python eva.py Outputs/Qwen3-0.6B/Qwen3-0.6B.jsonl --summary-only
```

Expected summary:

```text
| File | N | Correct | Invalid | Accuracy |
|---|---:|---:|---:|---:|
| ALL | 6112 | 1748 | 1405 | 0.2860 |
```

Run inference with a local Hugging Face model:

```bash
python infer.py --model-path /path/to/model --model-name model-name
```

Model-specific adapter settings are defined in `configs/models.json`.

## Limitations

- The 64-token generation cap provides a consistent evaluation budget, but it may penalize models that produce verbose reasoning before the final answer.
- Chat templates are used as interface adapters for instruction/chat models; they are not intended as model-specific task prompts.
- The benchmark measures multiple-choice answer extraction under a fixed protocol and should not be interpreted as evidence of clinical reliability.
- Dataset files are excluded from this repository. Users are responsible for obtaining and using MedQA under the applicable license terms.

## Citation

If this benchmark artifact is useful, please cite this repository using [`CITATION.cff`](CITATION.cff). This benchmark uses the MedQA dataset:

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
