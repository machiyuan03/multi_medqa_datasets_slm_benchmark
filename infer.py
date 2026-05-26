import argparse
import json
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Tuple

import torch
from tqdm.auto import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.generation import GenerationMixin
except ImportError:  # pragma: no cover - compatibility with older transformers layouts
    from transformers.generation.utils import GenerationMixin


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = Path.home() / "datasets" / "MedQA" / "questions"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Outputs"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "models.json"
DEFAULT_DATASET_CONFIG = PROJECT_ROOT / "configs" / "datasets" / "medqa_usmle.json"


PROMPT_EN = """Answer from the stem and options.

Question:
{question}

Options:
{options_text}

Required output format:
Final Answer: <option letter>
Mechanism reasoning: at most 3 bullets; one medical causal point per bullet; no stem restatement; no option-by-option analysis.
"""

PROMPT_ZH = """仅根据题干和选项作答。

题目：
{question}

选项：
{options_text}

要求输出格式：
Final Answer: <选项字母>
Mechanism reasoning: 最多 3 个要点；每个要点只写一个医学因果点；不要重复题干；不要逐个分析选项。
"""

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
FINAL_LINE_RE = re.compile(
    r"^\s*Final\s*Answer\s*[:：]\s*[<（(]?\s*(?:选项|option)?\s*([A-E])\s*[>）)]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
FINAL_RE = re.compile(
    r"Final\s*Answer\s*[:：]\s*[<（(]?\s*(?:选项|option)?\s*([A-E])\s*[>）)]?",
    re.IGNORECASE,
)
STANDALONE_OPTION_RE = re.compile(r"^\s*([A-E])\s*[\.、\)]?\s*$", re.IGNORECASE | re.MULTILINE)
LEADING_OPTION_RE = re.compile(r"^\s*([A-E])\s*[\.、\)]\s+\S", re.IGNORECASE | re.MULTILINE)


class GenerationResult(NamedTuple):
    full_text: str
    generated_text: str
    generated_token_count: int
    stopped_by_eos: bool


def load_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_model_configs(path: str) -> Dict[str, Dict]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)
    if not isinstance(configs, dict):
        raise ValueError(f"Model config file must contain a JSON object: {config_path}")
    return configs


def load_dataset_config(path: str | None) -> Dict:
    if not path:
        return {}
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Dataset config file must contain a JSON object: {config_path}")
    return config


def get_model_config(configs: Dict[str, Dict], model_name: str, model_path: str) -> Tuple[str, Dict]:
    candidates = [model_name, Path(str(model_path)).name, str(model_path)]
    for key in candidates:
        value = configs.get(key)
        if isinstance(value, dict):
            return key, value
    return "", {}


def is_test_file(fp: Path) -> bool:
    name = fp.name.lower()
    parts = {part.lower() for part in fp.parts}
    return name == "test.jsonl" or name.endswith("test.jsonl") or "test" in parts


def iter_input_files(path: str, canonical_test_files: List[str] | None = None) -> List[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)] if is_test_file(p) else []

    if canonical_test_files:
        canonical_files = [p / rel_path for rel_path in canonical_test_files]
        missing_files = [str(fp) for fp in canonical_files if not fp.exists()]
        if missing_files:
            raise FileNotFoundError(f"Canonical test files missing under {p}: {missing_files}")
        return [str(fp) for fp in canonical_files]

    files = sorted(p.rglob("*.jsonl"))
    return [str(fp) for fp in files if is_test_file(fp)]


def infer_language(question: str, input_file: str) -> str:
    zh_markers = ["/Mainland/", "\\Mainland\\", "/Taiwan/", "\\Taiwan\\"]
    if is_chinese_text(question) or any(token in input_file for token in zh_markers):
        return "zh"
    return "en"


def is_chinese_text(text: str) -> bool:
    return bool(CJK_RE.search(text or ""))


def extract_options(example: Dict):
    options = example.get("options", {})

    if isinstance(options, str):
        try:
            options = json.loads(options)
        except json.JSONDecodeError:
            return None

    if isinstance(options, list):
        normalized = {}
        for idx, item in enumerate(options):
            if isinstance(item, dict):
                key = item.get("key") or item.get("label") or item.get("option")
                value = item.get("value") or item.get("text") or item.get("content")
                if key is not None and value is not None:
                    normalized[str(key).strip()] = value
                elif value is not None:
                    normalized[chr(ord("A") + idx)] = value
            elif isinstance(item, str):
                normalized[chr(ord("A") + idx)] = item
        options = normalized

    if not isinstance(options, dict):
        return None

    ordered_keys = sorted(options.keys())
    if not ordered_keys:
        return None

    return {str(k).strip(): options[k] for k in ordered_keys}


def format_options_text(options: Dict[str, str]) -> str:
    return "\n".join(f"{k}. {v}" for k, v in options.items())


def shuffle_options(options: Dict[str, str], answer: str, rng: random.Random) -> Tuple[Dict[str, str], str, Dict[str, str]]:
    items = list(options.items())
    rng.shuffle(items)

    shuffled_options = {}
    option_key_map = {}
    for idx, (old_key, value) in enumerate(items):
        new_key = chr(ord("A") + idx)
        shuffled_options[new_key] = value
        option_key_map[str(old_key).strip().upper()] = new_key

    normalized_answer = str(answer or "").strip().upper()
    shuffled_answer = option_key_map.get(normalized_answer, "")
    return shuffled_options, shuffled_answer, option_key_map


def build_prompt(question: str, options_text: str, language: str) -> str:
    if language == "zh":
        return PROMPT_ZH.format(question=question, options_text=options_text)
    return PROMPT_EN.format(question=question, options_text=options_text)


def apply_chat_template(tokenizer, prompts: List[str], template_kwargs: Dict | None = None) -> List[str]:
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("Tokenizer does not provide a chat template.")
    template_kwargs = template_kwargs or {}
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        for prompt in prompts
    ]


def extract_option_from_output(output: str) -> str:
    if not output:
        return ""

    matches = FINAL_LINE_RE.findall(output)
    if matches:
        return matches[-1].upper()

    matches = FINAL_RE.findall(output)
    if matches:
        return matches[-1].upper()

    tail = output.splitlines()[-8:]
    tail_text = "\n".join(tail)
    matches = STANDALONE_OPTION_RE.findall(tail_text)
    if matches:
        return matches[-1].upper()

    leading_option_matches = LEADING_OPTION_RE.findall(output)
    if len(leading_option_matches) == 1:
        return leading_option_matches[0].upper()

    return ""


def parse_output(output: str) -> Tuple[str, str]:
    extracted_option = extract_option_from_output(output)
    if FINAL_LINE_RE.search(output) or FINAL_RE.search(output):
        return extracted_option, "matched_final"
    if extracted_option:
        return extracted_option, "matched_fallback"
    return "", "empty"


def load_local_model(
    model_path: str,
    use_4bit: bool = False,
    padding_side: str = "left",
    pad_token_fallback_to_eos: bool = True,
    attn_implementation: str | None = None,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required but not available.")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token_id is None and pad_token_fallback_to_eos:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = dict(trust_remote_code=True, device_map={"": 0}, torch_dtype=torch.float32)
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    if use_4bit:
        raise ValueError("--use-4bit is disabled because this script now requires full precision.")

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model_param = next(model.parameters())
    if model_param.dtype != torch.float32:
        model.to(dtype=torch.float32)
        model_param = next(model.parameters())
    if model_param.dtype != torch.float32:
        raise RuntimeError(f"Model loaded with dtype={model_param.dtype}; full precision torch.float32 is required.")
    if not hasattr(model, "generate"):
        model.generate = GenerationMixin.generate.__get__(model, type(model))
    model.eval()
    return tokenizer, model


def write_log(log_path: Path, lines: Iterable[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")


def count_jsonl(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def gpu_memory_report() -> str:
    if not torch.cuda.is_available():
        return "cuda_unavailable"
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    free = total - reserved
    return f"allocated={allocated:.0f}MB reserved={reserved:.0f}MB free_est={free:.0f}MB total={total:.0f}MB"


@torch.inference_mode()
def call_model(tokenizer, model, prompts: List[str], max_new_tokens: int = 128, use_cache: bool | None = None) -> List[GenerationResult]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_width = inputs["input_ids"].shape[-1]
    generate_kwargs = dict(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    if use_cache is not None:
        generate_kwargs["use_cache"] = use_cache
    output_ids = model.generate(**generate_kwargs)
    full_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    eos_token_ids = tokenizer.eos_token_id
    if eos_token_ids is None:
        eos_token_ids = []
    elif isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    results: List[GenerationResult] = []
    for full_text, token_ids in zip(full_texts, output_ids):
        generated_ids = token_ids[input_width:].detach().cpu().tolist()
        eos_positions = [idx for idx, token_id in enumerate(generated_ids) if token_id in eos_token_ids]
        stopped_by_eos = bool(eos_positions)
        if stopped_by_eos:
            generated_token_count = eos_positions[0] + 1
            generated_ids_for_decode = generated_ids[: eos_positions[0]]
        else:
            generated_token_count = len(generated_ids)
            generated_ids_for_decode = generated_ids
        generated_text = tokenizer.decode(generated_ids_for_decode, skip_special_tokens=True)
        results.append(GenerationResult(full_text, generated_text, generated_token_count, stopped_by_eos))

    return results


def benchmark_batch_size(
    tokenizer,
    model,
    sample_prompts: List[str],
    max_new_tokens: int,
    use_cache: bool | None = None,
    candidate_sizes: List[int] | None = None,
    slowdown_stop_ratio: float = 0.25,
) -> int:
    if not sample_prompts:
        return 1

    base_prompt = max(sample_prompts, key=len)
    probe_max_new_tokens = max(4, min(max_new_tokens, 16))
    candidate_sizes = candidate_sizes or [1, 2, 4, 8, 16, 32]
    max_probe_batch = 64
    best_size = 1
    best_throughput = 0.0
    best_latency = float("inf")

    def measure(n: int) -> Tuple[bool, float, float]:
        prompts = [base_prompt] * n
        print(f"[bench] probe size={n} probe_max_new_tokens={probe_max_new_tokens} {gpu_memory_report()}", flush=True)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        try:
            start.record()
            _ = call_model(tokenizer, model, prompts, max_new_tokens=probe_max_new_tokens, use_cache=use_cache)
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = float(start.elapsed_time(end))
            throughput = (n * 1000.0) / max(elapsed_ms, 1e-6)
            latency_ms = elapsed_ms / n
            print(
                f"[bench] ok size={n} elapsed_ms={elapsed_ms:.1f} latency_ms={latency_ms:.1f} throughput={throughput:.2f}/s {gpu_memory_report()}",
                flush=True,
            )
            return True, throughput, latency_ms
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[bench] OOM size={n} {gpu_memory_report()}", flush=True)
            return False, 0.0, float("inf")
        finally:
            torch.cuda.empty_cache()

    print(f"[bench] start max_probe_batch={max_probe_batch} {gpu_memory_report()}", flush=True)
    for size in candidate_sizes:
        if size > max_probe_batch:
            break
        ok, throughput, latency_ms = measure(size)
        if not ok:
            if best_throughput > 0:
                print(f"[bench] early_stop size={size} reason=oom best_size={best_size}", flush=True)
                break
            continue
        if throughput > best_throughput or (throughput == best_throughput and latency_ms < best_latency):
            best_size = size
            best_throughput = throughput
            best_latency = latency_ms
        elif best_throughput > 0 and throughput < best_throughput * slowdown_stop_ratio:
            print(
                f"[bench] early_stop size={size} reason=slowdown throughput={throughput:.2f}/s "
                f"best_size={best_size} best_throughput={best_throughput:.2f}/s",
                flush=True,
            )
            break

    print(
        f"[bench] selected_batch_size={best_size} best_throughput={best_throughput:.2f}/s best_latency_ms={best_latency:.1f} {gpu_memory_report()}",
        flush=True,
    )
    return max(1, best_size)


def run_batch_with_oom_fallback(tokenizer, model, prompts: List[str], max_new_tokens: int, batch_size: int, log_path: Path | None = None, use_cache: bool | None = None) -> List[GenerationResult]:
    if not prompts:
        return []

    outputs: List[GenerationResult] = []
    start = 0
    while start < len(prompts):
        end = min(start + batch_size, len(prompts))
        while True:
            try:
                outputs.extend(call_model(tokenizer, model, prompts[start:end], max_new_tokens=max_new_tokens, use_cache=use_cache))
                start = end
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                print(f"[batch] OOM fallback reduce batch_size={batch_size} {gpu_memory_report()}", flush=True)
                if log_path is not None:
                    write_log(log_path, [f"[batch] OOM fallback reduce batch_size={batch_size} {gpu_memory_report()}"])
                end = min(start + batch_size, len(prompts))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Input JSONL file or directory; defaults to the selected dataset config")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--model-path", required=True, help="Local model path or Hugging Face model name")
    parser.add_argument("--model-name", default=None, help="Override output file name; defaults to model path name")
    parser.add_argument("--use-4bit", action="store_true", help="Deprecated; full precision on GPU is required")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Maximum tokens to generate")
    parser.add_argument("--log-dir", default=str(PROJECT_ROOT / "Log"), help="Directory to write runtime logs")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick testing")
    parser.add_argument("--test-mode", action="store_true", help="Run a 100-sample diagnostic pass with larger generation output")
    parser.add_argument("--test-limit", type=int, default=100, help="Default sample limit when --test-mode is enabled")
    parser.add_argument("--test-max-new-tokens", type=int, default=256, help="Default max_new_tokens when --test-mode is enabled")
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG), help=f"Model adapter config JSON (default: {DEFAULT_MODEL_CONFIG})")
    parser.add_argument("--dataset-config", default=str(DEFAULT_DATASET_CONFIG), help=f"Dataset config JSON (default: {DEFAULT_DATASET_CONFIG})")
    parser.add_argument("--dataset-name", default=None, help="Override dataset output folder name; defaults to dataset config slug")
    parser.add_argument("--shuffle-options-seed", type=int, default=123, help="Shuffle options with this seed and remap answer labels")
    parser.add_argument("--chat-template", action="store_true", help="Wrap each prompt with the tokenizer chat template before generation")
    parser.add_argument("--no-chat-template", action="store_true", help="Disable chat template even if enabled in the model config")
    parser.add_argument("--batch-size", type=int, default=None, help="Override configured or benchmarked batch size")
    parser.add_argument("--resume", action="store_true", help="Append to an existing output file and skip records already written")
    args = parser.parse_args()

    dataset_config = load_dataset_config(args.dataset_config)
    dataset_slug = args.dataset_name or dataset_config.get("slug") or "dataset"
    dataset_display_name = dataset_config.get("display_name") or dataset_slug
    default_input = dataset_config.get("default_input") or str(DEFAULT_INPUT)
    input_path = Path(args.input or default_input).expanduser()
    canonical_test_files = [Path(rel_path) for rel_path in dataset_config.get("canonical_test_files", [])]
    expected_rows = dataset_config.get("expected_rows")
    config_shuffle_seed = dataset_config.get("shuffle_options_seed")
    if config_shuffle_seed is not None and args.shuffle_options_seed == parser.get_default("shuffle_options_seed"):
        args.shuffle_options_seed = int(config_shuffle_seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir)
    model_path = args.model_path
    log_path = log_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{Path(str(model_path)).name.replace('/', '_')}.log"
    model_name = args.model_name or Path(str(model_path)).name.replace('/', '_')
    model_configs = load_model_configs(args.model_config)
    model_config_key, model_config = get_model_config(model_configs, model_name, model_path)
    config_use_chat_template = bool(model_config.get("use_chat_template", False))
    effective_chat_template = True if args.chat_template else False if args.no_chat_template else config_use_chat_template
    chat_template_kwargs = model_config.get("chat_template_kwargs", {})
    if not isinstance(chat_template_kwargs, dict):
        raise ValueError(f"chat_template_kwargs must be an object for model config {model_config_key or model_name}")
    required_transformers_prefix = model_config.get("required_transformers_prefix")
    if required_transformers_prefix and not transformers.__version__.startswith(str(required_transformers_prefix)):
        raise RuntimeError(
            f"{model_name} requires transformers {required_transformers_prefix}.*, "
            f"but current version is {transformers.__version__}."
        )
    padding_side = str(model_config.get("padding_side", "left"))
    pad_token_fallback_to_eos = bool(model_config.get("pad_token_fallback_to_eos", True))
    mode = "test" if args.test_mode else "inference"
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else args.test_max_new_tokens if args.test_mode else 64
    configured_use_cache = model_config.get("use_cache")
    effective_use_cache = None if configured_use_cache is None else bool(configured_use_cache)
    effective_limit = args.limit
    if args.test_mode and effective_limit is None:
        effective_limit = args.test_limit
    output_suffix = ".chat.test.jsonl" if args.test_mode and effective_chat_template else ".test.jsonl" if args.test_mode else ".jsonl"
    out_path = output_dir / dataset_slug / model_name / f"{model_name}{output_suffix}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_local_model(
        model_path,
        use_4bit=args.use_4bit,
        padding_side=padding_side,
        pad_token_fallback_to_eos=pad_token_fallback_to_eos,
        attn_implementation=model_config.get("attn_implementation"),
    )
    if "use_cache" in model_config and hasattr(model, "config"):
        model.config.use_cache = bool(model_config["use_cache"])
    if not any(param.is_cuda for param in model.parameters()):
        raise RuntimeError("Model did not load onto GPU.")

    model_param = next(model.parameters())
    input_files = list(iter_input_files(str(input_path), canonical_test_files))
    if not input_files:
        raise RuntimeError("No test JSONL files matched under the input path.")

    file_stats = []
    total_examples = 0
    total_answered = 0
    option_len_counter = Counter()
    option_key_counter = Counter()

    for input_file in input_files:
        examples = list(load_jsonl(input_file))
        total_examples += len(examples)
        answered = 0
        lengths = Counter()
        keys = Counter()
        for ex in examples:
            opts = extract_options(ex)
            if opts is None:
                continue
            answered += 1
            total_answered += 1
            lengths[len(opts)] += 1
            option_len_counter[len(opts)] += 1
            keys.update(opts.keys())
            option_key_counter.update(opts.keys())
        file_stats.append((input_file, len(examples), answered, dict(lengths), dict(keys)))

    if expected_rows is not None and not args.test_mode and effective_limit is None and total_answered != int(expected_rows):
        raise RuntimeError(
            f"Dataset row count mismatch for {dataset_slug}: expected_rows={expected_rows}, parseable_options={total_answered}."
        )

    print(
        f"[dataset] name={dataset_display_name} slug={dataset_slug} mode={mode} "
        f"files={len(input_files)} total_examples={total_examples} parseable_options={total_answered}",
        flush=True,
    )
    print(f"[dataset] option_len_counts={dict(option_len_counter)} option_key_counts={dict(option_key_counter)}", flush=True)
    print(f"[dataset] selected_test_files={len(input_files)} total_test_examples={total_examples} parseable_options={total_answered}", flush=True)

    write_log(log_path, [
        f"[start] {datetime.now().isoformat(timespec='seconds')}",
        f"[input] {input_path}",
        f"[dataset_config] path={args.dataset_config} slug={dataset_slug} name={dataset_display_name}",
        f"[dataset_expected_rows] {expected_rows}",
        f"[mode] {mode}",
        f"[output] {out_path}",
        f"[model] {model_path}",
        f"[model_config] path={args.model_config} key={model_config_key or 'none'}",
        f"[max_new_tokens] {max_new_tokens}",
        f"[limit] {effective_limit}",
        f"[chat_template] {effective_chat_template}",
        f"[chat_template_kwargs] {chat_template_kwargs}",
        f"[shuffle_options_seed] {args.shuffle_options_seed}",
        f"[use_cache] {effective_use_cache}",
        f"[padding_side] {tokenizer.padding_side}",
        f"[pad_token] id={tokenizer.pad_token_id} text={tokenizer.pad_token!r}",
        f"[eos_token] id={tokenizer.eos_token_id} text={tokenizer.eos_token!r}",
        f"[device] cuda_available={torch.cuda.is_available()} model_device={model_param.device} dtype={model_param.dtype}",
        f"[dataset] files={len(input_files)} total_examples={total_examples} parseable_options={total_answered}",
        f"[dataset] option_len_counts={dict(option_len_counter)}",
        f"[dataset] option_key_counts={dict(option_key_counter)}",
    ])
    for fp, n, answered, lengths, keys in file_stats:
        write_log(log_path, [f"[file] {fp} examples={n} parseable_options={answered} option_len_counts={lengths} option_key_counts={keys}"])

    option_shuffle_rng = random.Random(args.shuffle_options_seed)
    all_records = []
    for input_file in input_files:
        for example in load_jsonl(input_file):
            opts = extract_options(example)
            if opts is None:
                continue
            original_answer = str(example.get("answer_idx", "") or "").strip().upper()
            shuffled_opts, shuffled_answer, option_key_map = shuffle_options(opts, original_answer, option_shuffle_rng)
            all_records.append((input_file, example.get("question", ""), opts, original_answer, shuffled_opts, shuffled_answer, option_key_map))

    total_to_process = min(len(all_records), effective_limit) if effective_limit is not None else len(all_records)
    resume_start = 0
    if args.resume and out_path.exists():
        resume_start = count_jsonl(str(out_path))
        if resume_start > total_to_process:
            raise RuntimeError(f"Cannot resume: existing rows ({resume_start}) exceed target rows ({total_to_process}).")
        all_records = all_records[resume_start:total_to_process]
        total_to_process = len(all_records)
        print(f"[resume] existing_rows={resume_start} remaining={total_to_process}", flush=True)
        write_log(log_path, [f"[resume] existing_rows={resume_start} remaining={total_to_process}"])
    print(f"[progress] total_to_process={total_to_process} test_total={len(all_records)}", flush=True)

    if total_to_process == 0:
        write_log(log_path, ["[summary] processed=0 skipped=0", f"[end] {datetime.now().isoformat(timespec='seconds')}"])
        return

    probe_records = all_records[: min(8, total_to_process)]
    probe_prompts = [build_prompt(q, format_options_text(shuffled_opts), infer_language(q, src)) for src, q, _, _, shuffled_opts, _, _ in probe_records]
    probe_model_inputs = apply_chat_template(tokenizer, probe_prompts, chat_template_kwargs) if effective_chat_template else probe_prompts
    configured_batch_size = args.batch_size if args.batch_size is not None else model_config.get("batch_size")
    if configured_batch_size is not None:
        stable_batch_size = max(1, int(configured_batch_size))
        print(f"[batch] configured_batch_size={stable_batch_size} {gpu_memory_report()}", flush=True)
        write_log(log_path, [f"[batch] configured_batch_size={stable_batch_size} {gpu_memory_report()}"])
    else:
        stable_batch_size = benchmark_batch_size(tokenizer, model, probe_model_inputs, max_new_tokens=max_new_tokens, use_cache=effective_use_cache)
        print(f"[batch] benchmark_batch_size={stable_batch_size} {gpu_memory_report()}", flush=True)
        write_log(log_path, [f"[batch] benchmark_batch_size={stable_batch_size} {gpu_memory_report()}"])

    processed = 0
    skipped = 0
    progress = tqdm(total=total_to_process, desc="Inference", unit="sample")

    open_mode = "a" if args.resume and resume_start > 0 else "w"
    with open(out_path, open_mode, encoding="utf-8") as writer:
        try:
            for batch_start in range(0, total_to_process, stable_batch_size):
                batch_end = min(batch_start + stable_batch_size, total_to_process)
                batch_records = all_records[batch_start:batch_end]
                batch_prompts = [build_prompt(q, format_options_text(shuffled_opts), infer_language(q, src)) for src, q, _, _, shuffled_opts, _, _ in batch_records]
                batch_model_inputs = apply_chat_template(tokenizer, batch_prompts, chat_template_kwargs) if effective_chat_template else batch_prompts
                print(f"[batch] start size={len(batch_prompts)} target={stable_batch_size} {gpu_memory_report()}", flush=True)
                outputs = run_batch_with_oom_fallback(tokenizer, model, batch_model_inputs, max_new_tokens=max_new_tokens, batch_size=stable_batch_size, log_path=log_path, use_cache=effective_use_cache)
                print(f"[batch] done size={len(batch_prompts)} target={stable_batch_size} {gpu_memory_report()}", flush=True)

                for (src_file, q, original_opts, original_ans, shuffled_opts, ans, option_key_map), result in zip(batch_records, outputs):
                    extracted_option, parse_status = parse_output(result.generated_text)
                    prompt = build_prompt(q, format_options_text(shuffled_opts), infer_language(q, src_file))
                    record = {
                        "dataset": dataset_slug,
                        "dataset_name": dataset_display_name,
                        "dataset_config": args.dataset_config,
                        "source_file": src_file,
                        "question": q,
                        "options": shuffled_opts,
                        "answer": ans,
                        "original_options": original_opts,
                        "original_answer": original_ans,
                        "option_key_map": option_key_map,
                        "option_shuffle_seed": args.shuffle_options_seed,
                        "prompt": prompt,
                        "model_output": result.full_text,
                        "generated_output": result.generated_text,
                        "generated_token_count": result.generated_token_count,
                        "max_new_tokens": max_new_tokens,
                        "model_config_key": model_config_key,
                        "chat_template": effective_chat_template,
                        "padding_side": tokenizer.padding_side,
                        "pad_token_id": tokenizer.pad_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                        "stopped_by_eos": result.stopped_by_eos,
                        "possibly_truncated": result.generated_token_count >= max_new_tokens and not result.stopped_by_eos,
                        "extracted_option": extracted_option,
                        "parse_status": parse_status,
                        "parse_source": "generated_output",
                    }
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    writer.flush()
                    processed += 1
                    progress.update(1)
                    if processed % 50 == 0 or processed == total_to_process:
                        progress.set_postfix(processed=processed, skipped=skipped, batch=stable_batch_size)
                        write_log(log_path, [f"[progress] processed={processed}/{total_to_process} skipped={skipped} batch={stable_batch_size}"])
                    if effective_limit is not None and processed >= effective_limit:
                        write_log(log_path, [f"[summary] processed={processed} skipped={skipped} limit={effective_limit}"])
                        return
        finally:
            progress.close()

    write_log(log_path, [f"[summary] processed={processed} skipped={skipped}", f"[end] {datetime.now().isoformat(timespec='seconds')}"])


if __name__ == "__main__":
    main()
