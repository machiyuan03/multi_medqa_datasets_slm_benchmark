# Released Benchmark Outputs

This directory contains the selected full-run JSONL outputs used in the public result table. Each retained model output contains 6,112 records and excludes diagnostic or test-mode generations.

## Files

| Model | Output |
| --- | --- |
| Qwen3.5-0.8B | `Qwen3.5-0.8B-fp32/Qwen3.5-0.8B-fp32.jsonl` |
| Qwen2.5-0.5B-Instruct | `Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct.jsonl` |
| Qwen3-0.6B | `Qwen3-0.6B/Qwen3-0.6B.jsonl` |
| Granite 4.0 350M | `granite-4.0-350m/granite-4.0-350m.jsonl` |
| Gemma 3 270M IT | `gemma-3-270m-it-ct/gemma-3-270m-it-ct.jsonl` |
| SmolLM2-360M-Instruct | `SmolLM2-360M-Instruct/SmolLM2-360M-Instruct.jsonl` |
| OLMo-2-0425-1B-Instruct | `OLMo-2-0425-1B-Instruct/OLMo-2-0425-1B-Instruct.jsonl` |
| Llama 3.2 1B Instruct | `Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct.jsonl` |
| MiniCPM4-0.5B | `MiniCPM4-0.5B/MiniCPM4-0.5B.jsonl` |
| LFM2-350M | `LFM2-350M/LFM2-350M.jsonl` |

## JSONL Schema

Each line is one evaluated example. Important fields include:

- `source_file`: original test split path used during generation.
- `question`: question stem.
- `options`: shuffled answer options presented to the model.
- `answer`: correct answer after option shuffling.
- `original_options`: options before shuffling.
- `original_answer`: original dataset answer label before shuffling.
- `option_key_map`: mapping from original option labels to shuffled labels.
- `prompt`: benchmark prompt before optional chat-template wrapping.
- `model_output`: decoded prompt plus generated continuation.
- `generated_output`: generated continuation used for evaluation.
- `extracted_option`: answer label extracted by the shared parser.
- `parse_status`: parser status (`matched_final`, `matched_fallback`, or `empty`).
- `max_new_tokens`: generation cap used for the run.
- `possibly_truncated`: whether generation reached the token cap without EOS.

The `answer` field is the shuffled-label target used for scoring. Use `original_answer` and `option_key_map` only when auditing the option remapping.
