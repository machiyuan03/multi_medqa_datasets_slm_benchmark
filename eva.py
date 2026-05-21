import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


LABELS = ("A", "B", "C", "D", "E")


def iter_jsonl_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.jsonl"))


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


def compute_metrics(records: Iterable[Dict]) -> Dict[str, float | int]:
    total = 0
    correct = 0
    invalid_predictions = 0

    for record in records:
        y_true = normalize_label(record.get("answer"))
        y_pred = normalize_label(record.get("extracted_option"))
        if not y_true:
            continue

        total += 1
        if not y_pred:
            invalid_predictions += 1
            continue

        if y_pred == y_true:
            correct += 1

    return {
        "total": total,
        "correct": correct,
        "invalid_predictions": invalid_predictions,
        "accuracy": correct / total if total else 0.0,
    }


def format_float(value: float) -> str:
    return f"{value:.4f}"


def print_table(rows: List[Dict]) -> None:
    print("| File | N | Correct | Invalid | Accuracy |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            "| {file} | {total} | {correct} | {invalid_predictions} | {accuracy} |".format(
                file=row["file"],
                total=row["total"],
                correct=row["correct"],
                invalid_predictions=row["invalid_predictions"],
                accuracy=format_float(row["accuracy"]),
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate infer.py JSONL outputs.")
    parser.add_argument("path", help="Output JSONL file or directory containing JSONL files")
    parser.add_argument("--summary-only", action="store_true", help="Only print aggregated metrics across all files")
    args = parser.parse_args()

    input_path = Path(args.path)
    files = iter_jsonl_files(input_path)
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


if __name__ == "__main__":
    main()
