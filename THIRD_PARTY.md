# Third-Party Resources

This repository contains benchmark code, configuration files, and reported aggregate results. It does not redistribute upstream datasets or model weights.

Users are responsible for obtaining datasets and models from their original providers and complying with the corresponding licenses, terms of use, access restrictions, and citation requirements.

## Datasets

The benchmark uses canonical evaluation subsets from the following resources:

| Dataset | Source / provider | Notes |
| --- | --- | --- |
| MedQA / USMLE | MedQA | Mainland, Taiwan, and US test files |
| MedMCQA | `awinml/medmcqa`; MedMCQA official resources | Public labeled validation split |
| PubMedQA | PubMedQA official repository / PQA-L | Official PQA-L test set |
| MedXpertQA | `TsinghuaC3I/MedXpertQA` | Text split only; multimodal split excluded |
| MMLU medical subsets | `cais/mmlu` | Six medical or biomedical test subjects |
| MMedBench | `Henrychur/MMedBench`; MMedLM resources | Official multilingual test split |
| AfriMedQA | `afrimedqa/afrimedqa_v2` | Expert MCQ test subset |
| CRAFT-MedQA | `ingoziegler/CRAFT-MedQA` | XL variant |

## Models

The benchmark reports results for the following model families or checkpoints:

- Qwen3.5-0.8B
- Qwen2.5-0.5B-Instruct
- Qwen3-0.6B
- Granite 4.0 350M
- Gemma 3 270M IT
- SmolLM2-360M-Instruct
- OLMo-2-0425-1B-Instruct
- Llama 3.2 1B Instruct
- MiniCPM4-0.5B
- LFM2-350M

Model weights are not included in this repository. Some models may require accepting provider-specific licenses or gated access terms.

## Citation Responsibility

When publishing results derived from this repository, cite this repository as well as the original dataset and model sources. The aggregate benchmark results here do not replace the citation requirements of upstream resources.
