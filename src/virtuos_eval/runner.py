"""Run items through a provider and write a scorecard.

The runner is resumable. Each result is appended to ``results.jsonl`` as it
arrives, and a rerun in the same directory skips items that already have an
answer. Failed calls are recorded with ``answer: null`` and retried on the next
run, so a rate limit does not cost the earlier work.
"""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from .providers import Provider, parse_letter
from .scenarios import Item

FIELDS = [
    "item_id",
    "entry_id",
    "virtue",
    "layer",
    "tradition",
    "correct",
    "answer",
    "ok",
    "error",
    "raw",
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "run"


def _row(item: Item, raw: str) -> dict:
    answer = parse_letter(raw)
    return {
        "item_id": item.item_id,
        "entry_id": item.entry_id,
        "virtue": item.virtue,
        "layer": item.layer,
        "tradition": item.tradition,
        "correct": item.correct,
        "answer": answer,
        "ok": answer == item.correct,
        "error": None,
        "raw": raw[:200],
    }


def _error_row(item: Item, error: Exception) -> dict:
    return {
        "item_id": item.item_id,
        "entry_id": item.entry_id,
        "virtue": item.virtue,
        "layer": item.layer,
        "tradition": item.tradition,
        "correct": item.correct,
        "answer": None,
        "ok": None,
        "error": str(error)[:300],
        "raw": "",
    }


def _table(rows: list[dict], key: str) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row[key], []).append(row)
    table = []
    for name, group in buckets.items():
        graded = [r for r in group if r["ok"] is not None]
        table.append(
            {
                "name": name,
                "items": len(graded),
                "correct": sum(1 for r in graded if r["ok"]),
                "errors": sum(1 for r in group if r["ok"] is None),
            }
        )
    table.sort(key=lambda t: (-t["items"], t["name"]))
    return table


def evaluate(
    items: list[Item],
    provider: Provider,
    *,
    existing: dict[str, dict] | None = None,
    on_result=None,
    verbose: bool = False,
) -> dict:
    """Run items and return a summary. Writes nothing unless ``on_result`` is set."""
    existing = existing or {}
    rows: list[dict] = []

    for index, item in enumerate(items, 1):
        previous = existing.get(item.item_id)
        if previous is not None and previous.get("answer") is not None:
            rows.append(previous)
            continue

        try:
            raw = provider.complete(item.prompt)
            row = _row(item, raw)
        except Exception as exc:  # a failed call is data, not a crash
            row = _error_row(item, exc)

        rows.append(row)
        if on_result is not None:
            on_result(row)

        if verbose:
            mark = "ERR " if row["ok"] is None else ("ok  " if row["ok"] else "MISS")
            shown = row["error"] if row["ok"] is None else f"expected {item.correct}, got {row['answer']}"
            print(f"[{index}/{len(items)}] {mark} {item.virtue}: {shown}")

    graded = [r for r in rows if r["ok"] is not None]
    correct = sum(1 for r in graded if r["ok"])
    errors = len(rows) - len(graded)
    low, high = wilson(correct, len(graded))
    return {
        "items": len(rows),
        "graded": len(graded),
        "correct": correct,
        "errors": errors,
        "accuracy": (correct / len(graded)) if graded else 0.0,
        "wilson95": [low, high],
        "by_virtue": _table(rows, "virtue"),
        "by_layer": _table(rows, "layer"),
        "by_tradition": _table(rows, "tradition"),
        "results": rows,
    }


def run(
    items: list[Item],
    provider: Provider,
    *,
    out_dir: str | Path | None = None,
    model: str | None = None,
    corpus_version: str = "0.1-draft",
    design: str = "fit",
    label: str | None = None,
    verbose: bool = True,
    fresh: bool = False,
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if out_dir is None:
        out_dir = Path("runs") / f"{stamp}-{_slug(label or provider.name)}"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "items.jsonl").open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item.as_dict(), ensure_ascii=False) + "\n")

    results_path = out / "results.jsonl"
    existing: dict[str, dict] = {}
    if fresh:
        results_path.unlink(missing_ok=True)
    elif results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            existing[row["item_id"]] = row

    written = 0
    with results_path.open("a", encoding="utf-8") as fh:

        def on_result(row: dict) -> None:
            nonlocal written
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1

        summary = evaluate(items, provider, existing=existing, on_result=on_result, verbose=verbose)

    rows = summary["results"]
    with (out / "scores.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    stamped = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    low, high = summary["wilson95"]
    lines = [
        f"# VirtuOS scorecard: {label or provider.name}",
        "",
        f"- Provider: `{provider.name}`" + (f" (`{model}`)" if model else ""),
        f"- Design: `{design}`",
        f"- Corpus: `{corpus_version}` (draft, entries AI-drafted, no human labels)",
        f"- Run: {stamped}",
        f"- Items: {summary['items']}",
        f"- Graded: {summary['graded']}",
        f"- Correct: {summary['correct']} of {summary['graded']} ({summary['accuracy']:.1%})",
        f"- Errors: {summary['errors']} (excluded from accuracy, retried on rerun)",
        f"- Wilson 95% interval: {low:.1%} to {high:.1%}",
        "- Chance for three options: 33.3%",
        "",
        "This measures agreement with one drafted answer under a forced choice. It is",
        "not a measure of wisdom, and the corpus has not been reviewed. Treat it as a",
        "smoke test for the pipeline until the label layer exists.",
        "",
        "## By virtue",
        "",
        "| Virtue | Items | Correct | Errors | Accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["by_virtue"]:
        accuracy = row["correct"] / row["items"] if row["items"] else 0.0
        lines.append(
            f"| {row['name']} | {row['items']} | {row['correct']} | {row['errors']} | {accuracy:.0%} |"
        )

    lines += ["", "## By layer", "", "| Layer | Items | Correct | Errors | Accuracy |", "|---|---:|---:|---:|---:|"]
    for row in summary["by_layer"]:
        accuracy = row["correct"] / row["items"] if row["items"] else 0.0
        lines.append(
            f"| {row['name']} | {row['items']} | {row['correct']} | {row['errors']} | {accuracy:.0%} |"
        )

    lines += ["", "## By tradition (top 10)", "", "| Tradition | Items | Correct | Errors | Accuracy |", "|---|---:|---:|---:|---:|"]
    for row in summary["by_tradition"][:10]:
        accuracy = row["correct"] / row["items"] if row["items"] else 0.0
        lines.append(
            f"| {row['name']} | {row['items']} | {row['correct']} | {row['errors']} | {accuracy:.0%} |"
        )

    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {
        "provider": provider.name,
        "model": model,
        "design": design,
        "corpus_version": corpus_version,
        "run": stamped,
        "items": summary["items"],
        "graded": summary["graded"],
        "correct": summary["correct"],
        "errors": summary["errors"],
        "accuracy": summary["accuracy"],
        "wilson95": summary["wilson95"],
        "new_results": written,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    if verbose:
        print(
            f"\n{summary['correct']}/{summary['graded']} correct ({summary['accuracy']:.1%}), "
            f"95% CI {low:.1%} to {high:.1%}"
            + (f"; {summary['errors']} errors" if summary["errors"] else "")
        )
        print(f"Wrote {out}")
    return out
