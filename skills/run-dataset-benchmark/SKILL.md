---
name: run-dataset-benchmark
description: "Use this skill in the multi_medqa_datasets_slm_benchmark project when running one new medical QA dataset end to end: investigate dataset versions and structure, prepare dataset config, ask for human approval, dispatch one subagent per model, validate outputs, and update 实验日志.md."
---

# Run Dataset Benchmark

Use this project-local workflow for one dataset at a time. The main agent owns dataset investigation, benchmark口径, scheduling, validation, and log integration. Model execution is delegated: one model task goes to one subagent, and follow-up/refactor work should return to the same subagent that ran that model when possible.

## Start State

Before changing files, read `AGENTS.md`, `实验日志.md`, `configs/models.json`, existing `configs/datasets/*.json`, and relevant parts of `infer.py` / `eva.py`.

If the user does not name a dataset, use the leftmost unfinished dataset in the result table. Do not start model runs until the dataset investigation has been reported and the user has approved the exact version/split/口径.

## Dataset Investigation

Locate or download the dataset under `~/datasets/<DatasetName>/`. Investigate enough to answer:

- Source dataset name, local path, license/readme if available, and whether HF/default config is used.
- Available configs, versions, languages, splits, file names, and row counts.
- Which versions are canonical raw/eval data and which are processed, translated, augmented, duplicated, unlabeled, or otherwise unsuitable.
- Question fields, option fields/order, answer label format, explanation fields, IDs, images if any, and language.
- Whether train/test are usable for benchmark accuracy. Do not use unlabeled or contaminating splits; remove or ignore them only when the user explicitly approves.

Create or update `configs/datasets/<dataset-slug>.json` only after the canonical eval file and expected row count are clear. Prefer a canonical JSONL with stable fields such as `question`, `options`, `answer_idx`, and `metadata` if normalization is needed.

Prompt policy: reuse the existing unified prompt if the dataset structure supports it. If a dataset-specific prompt seems necessary, stop and ask for human approval before changing it. After approval, it must be one prompt for the whole dataset across all models, and the reason must be logged.

Report the investigation for human audit before dispatching models. The report should include the proposed canonical split/file, expected rows, prompt choice, any proposed prompt change, and discarded versions/splits with reasons.

## Dispatch Policy

After user approval, assign each `dataset x model` run to a subagent. Do not bundle multiple models into one subagent unless the user explicitly changes the rule. The main agent does not solve model-specific pipeline problems directly; it specifies the task, monitors, asks for reruns when needed, and validates deliverables.

Long runs should use `tmux`, with the session name and attach command reported immediately. Avoid concurrent GPU-heavy jobs unless the user has approved the resource plan.

Model settings should be HF/default or existing `configs/models.json` defaults. Minimal compatibility changes are allowed only when default behavior cannot produce effective, evaluable outputs. For every non-default change, log: why default failed, what changed, and what the change fixed.

## Per-Model Run Contract

Give each subagent a concrete contract:

```bash
python infer.py \
  --dataset-config configs/datasets/<dataset>.json \
  --model-path <model-path> \
  --model-name <model-name> \
  --test-mode --limit 1
```

If the smoke test is plausible, run full benchmark through `tmux` and save stdout/stderr under `Log/`:

```bash
tmux new -s <session> 'cd ~/projects/multi_medqa_datasets_slm_benchmark && python infer.py --dataset-config configs/datasets/<dataset>.json --model-path <model-path> --model-name <model-name> 2>&1 | tee Log/<session>.stdout.log'
```

Evaluate only the full output:

```bash
python eva.py Outputs/<dataset>/<model>/<model>.jsonl --summary-only
```

Diagnostic outputs such as `.test.jsonl` or limited runs are not final benchmark results.

## Validation

For each returned model result, the main agent verifies before writing the table:

- Full output path is `Outputs/<dataset>/<model>/<model>.jsonl`.
- Output row count equals the dataset `expected_rows`.
- The run log ended normally or has a documented recover/resume path.
- Evaluation summary is from the full output, not a test file.
- Parser/invalid statistics, answer distribution, truncation, EOS behavior, and repeated empty outputs are not obvious pipeline failures.
- If a suspicious result conflicts with prior dataset behavior, ask the responsible subagent to investigate and rerun or justify it before accepting.

If a model has high invalid rate but the pipeline is confirmed consistent with prior accepted behavior, accept the full benchmark result and write a short adaptation note instead of silently aborting.

## Log Writeback

After each accepted full result, update `实验日志.md` immediately. Keep the current log structure: the first two tables are hot memory and must stay compact; the notes after them are cold memory and should be concise, loaded only when needed.

1. First main table: write only final benchmark accuracy.
2. Second table: write only very short adaptation text using the current口径, such as `默认`, `--no-chat-template`, `resume batch=1`, or `数据集专用prompt`.
3. `# 适配笔记` and later sections: cold memory. Add brief details only when needed, especially for dataset-specific decisions, approved prompt changes, and non-default model changes.

Do not overwrite a previous final output unless the user approved replacement or the rerun is explicitly fixing the same dataset/model. When replacing, note the reason in the adaptation notes.

## Guardrails

- Benchmark fairness is higher priority than score.
- Do not change question semantics, answer mapping, or evaluation rules for one model.
- Do not create a separate inference script for one model/dataset.
- Do not run processed/derived dataset versions as separate benchmark entries without user approval.
- Preserve unrelated user or git changes.
