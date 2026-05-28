import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List


LABELS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")
ANSWER_TEXT_RE = r"([A-J](?:\s*(?:[,，;/、&]|\band\b)\s*[A-J])*)"
FINAL_LINE_RE = re.compile(
    rf"^\s*Final\s*Answer\s*[:：]\s*[<（(]?\s*(?:选项|option)?\s*{ANSWER_TEXT_RE}\s*[>）)]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
FINAL_RE = re.compile(
    rf"Final\s*Answer\s*[:：]\s*[<（(]?\s*(?:选项|option)?\s*{ANSWER_TEXT_RE}\s*[>）)]?",
    re.IGNORECASE,
)
STANDALONE_OPTION_RE = re.compile(rf"^\s*{ANSWER_TEXT_RE}\s*[\.、\)]?\s*$", re.IGNORECASE | re.MULTILINE)
LEADING_OPTION_RE = re.compile(r"^\s*([A-J])\s*[\.、\)]\s+\S", re.IGNORECASE | re.MULTILINE)
LENIENT_ANSWER_RE = re.compile(
    rf"(?:^|\n)\s*(?:[*_`]+\s*)?(?:the\s+)?(?:correct\s+)?answer\s*(?:is|[:：])\s*"
    rf"[<（(]?\s*(?:选项|option)?\s*{ANSWER_TEXT_RE}\s*[>）)]?",
    re.IGNORECASE,
)

FORMAL_EXCLUDED_PATH_MARKERS = ("diag", "backup", "rework", "chat-template-old", "overrun")


def is_formal_output_file(path: Path) -> bool:
    if path.suffix != ".jsonl":
        return False
    if path.stem != path.parent.name:
        return False
    path_text = "/".join(part.lower() for part in path.parts)
    return not any(marker in path_text for marker in FORMAL_EXCLUDED_PATH_MARKERS)


def iter_jsonl_files(path: Path, include_all: bool = False) -> List[Path]:
    if path.is_file():
        return [path]

    files = sorted(path.rglob("*.jsonl"))
    if include_all:
        return files
    return [file_path for file_path in files if is_formal_output_file(file_path)]


def load_records(path: Path) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc


def normalize_label(value) -> str:
    text = str(value or "").strip().upper()
    return text if text in LABELS else ""


def normalize_answer_labels(value) -> List[str]:
    if isinstance(value, list):
        raw_parts = [str(item) for item in value]
    else:
        raw_parts = re.split(r"[,，;/、&]|\band\b|\bor\b", str(value or ""), flags=re.IGNORECASE)

    labels = []
    for part in raw_parts:
        for label in re.findall(r"[A-J]", part.upper()):
            if label not in labels:
                labels.append(label)
    return [label for label in LABELS if label in set(labels)]


def format_answer_labels(labels: List[str]) -> str:
    return ",".join(label for label in LABELS if label in set(labels))


def extract_option_from_output(output: str, allow_multi_answer: bool = False) -> str:
    if not output:
        return ""

    matches = FINAL_LINE_RE.findall(output)
    if matches:
        labels = normalize_answer_labels(matches[-1])
        return format_answer_labels(labels if allow_multi_answer else labels[-1:])

    matches = FINAL_RE.findall(output)
    if matches:
        labels = normalize_answer_labels(matches[-1])
        return format_answer_labels(labels if allow_multi_answer else labels[-1:])

    tail = output.splitlines()[-8:]
    tail_text = "\n".join(tail)
    matches = STANDALONE_OPTION_RE.findall(tail_text)
    if matches:
        labels = normalize_answer_labels(matches[-1])
        return format_answer_labels(labels if allow_multi_answer else labels[-1:])

    leading_option_matches = LEADING_OPTION_RE.findall(output)
    if len(leading_option_matches) == 1:
        return leading_option_matches[0].upper()

    return ""


def extract_lenient_option_from_output(output: str, allow_multi_answer: bool = False) -> str:
    extracted_option = extract_option_from_output(output, allow_multi_answer=allow_multi_answer)
    if extracted_option:
        return extracted_option

    matches = LENIENT_ANSWER_RE.findall(output or "")
    if matches:
        labels = normalize_answer_labels(matches[-1])
        return format_answer_labels(labels if allow_multi_answer else labels[-1:])

    return ""


def prediction_from_record(record: Dict, lenient: bool = False) -> str:
    allow_multi_answer = bool(record.get("allow_multi_answer", False))
    if "generated_output" in record:
        output = str(record.get("generated_output") or "")
        if lenient:
            return extract_lenient_option_from_output(output, allow_multi_answer=allow_multi_answer)
        return extract_option_from_output(output, allow_multi_answer=allow_multi_answer)
    labels = normalize_answer_labels(record.get("extracted_options") or record.get("extracted_option"))
    return format_answer_labels(labels if allow_multi_answer else labels[-1:])


def compute_metrics(records: Iterable[Dict], lenient: bool = False) -> Dict[str, float | int]:
    total = 0
    correct = 0
    invalid_predictions = 0
    multi_answer_total = 0
    multi_answer_correct = 0

    for record in records:
        y_true_labels = normalize_answer_labels(record.get("answer_labels") or record.get("answer"))
        y_pred_labels = normalize_answer_labels(prediction_from_record(record, lenient=lenient))
        if not y_true_labels:
            continue

        total += 1
        if len(y_true_labels) > 1:
            multi_answer_total += 1
        if not y_pred_labels:
            invalid_predictions += 1
            continue

        if set(y_pred_labels) == set(y_true_labels):
            correct += 1
            if len(y_true_labels) > 1:
                multi_answer_correct += 1

    return {
        "total": total,
        "correct": correct,
        "invalid_predictions": invalid_predictions,
        "multi_answer_total": multi_answer_total,
        "multi_answer_correct": multi_answer_correct,
        "accuracy": correct / total if total else 0.0,
    }


def format_float(value: float) -> str:
    return f"{value:.4f}"


def print_table(rows: List[Dict]) -> None:
    print("| File | N | Correct | Invalid | Multi N | Multi Correct | Accuracy |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            "| {file} | {total} | {correct} | {invalid_predictions} | {multi_answer_total} | {multi_answer_correct} | {accuracy} |".format(
                file=row["file"],
                total=row["total"],
                correct=row["correct"],
                invalid_predictions=row["invalid_predictions"],
                multi_answer_total=row["multi_answer_total"],
                multi_answer_correct=row["multi_answer_correct"],
                accuracy=format_float(row["accuracy"]),
            )
        )


def print_lenient_diagnostic_table(rows: List[Dict]) -> None:
    print("| File | N | Strict Correct | Strict Invalid | Strict Accuracy | Lenient Correct | Lenient Invalid | Lenient Accuracy | Correct Gain |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        strict = row["strict"]
        lenient = row["lenient"]
        print(
            "| {file} | {total} | {strict_correct} | {strict_invalid} | {strict_accuracy} | {lenient_correct} | {lenient_invalid} | {lenient_accuracy} | {correct_gain} |".format(
                file=row["file"],
                total=strict["total"],
                strict_correct=strict["correct"],
                strict_invalid=strict["invalid_predictions"],
                strict_accuracy=format_float(strict["accuracy"]),
                lenient_correct=lenient["correct"],
                lenient_invalid=lenient["invalid_predictions"],
                lenient_accuracy=format_float(lenient["accuracy"]),
                correct_gain=lenient["correct"] - strict["correct"],
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate infer.py JSONL outputs.")
    parser.add_argument("path", help="Output JSONL file or directory containing JSONL files")
    parser.add_argument("--summary-only", action="store_true", help="Only print aggregated metrics across all files")
    parser.add_argument("--all-jsonl", action="store_true", help="Include every JSONL under a directory, including diagnostics and old/test outputs")
    parser.add_argument("--lenient-diagnostic", action="store_true", help="Print strict vs lenient parser diagnostics; strict remains the official score")
    args = parser.parse_args()

    input_path = Path(args.path).expanduser()
    files = iter_jsonl_files(input_path, include_all=args.all_jsonl)
    if not files:
        raise RuntimeError(f"No JSONL files found under {input_path}")

    rows = []
    all_records = []
    for file_path in files:
        records = list(load_records(file_path))
        metrics = compute_metrics(records)
        metrics["file"] = str(file_path)
        rows.append(metrics)
        all_records.extend(records)

    summary = compute_metrics(all_records)
    summary["file"] = "ALL"

    if args.summary_only:
        print_table([summary])
    else:
        print_table(rows + [summary])

    if args.lenient_diagnostic:
        diagnostic_rows = []
        for file_path in files:
            records = list(load_records(file_path))
            diagnostic_rows.append(
                {
                    "file": str(file_path),
                    "strict": compute_metrics(records),
                    "lenient": compute_metrics(records, lenient=True),
                }
            )
        diagnostic_rows.append(
            {
                "file": "ALL",
                "strict": compute_metrics(all_records),
                "lenient": compute_metrics(all_records, lenient=True),
            }
        )
        print()
        print_lenient_diagnostic_table([diagnostic_rows[-1]] if args.summary_only else diagnostic_rows)


if __name__ == "__main__":
    main()
