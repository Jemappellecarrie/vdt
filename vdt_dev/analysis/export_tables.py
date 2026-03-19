from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from vdt_dev.analysis._plotting import load_csv_rows, maybe_prepare_aggregate


def _format_value(value: Any, *, precision: int = 2) -> str:
    if value in {None, "", "nan", "None"}:
        return "-"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def _rows_to_markdown(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(header)) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def _rows_to_latex(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [
        "\\begin{tabular}{" + "l" * len(headers) + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_format_value(row.get(header)) for header in headers) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def export_tables(
    *,
    summary_csv: str | None = None,
    output_roots: List[str] | None = None,
    aggregate_dir: str = "vdt_dev/outputs/aggregated",
    output_dir: str = "vdt_dev/paper_tables",
) -> Dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if summary_csv:
        rows = load_csv_rows(summary_csv)
    else:
        summary = maybe_prepare_aggregate(output_roots, destination_dir=aggregate_dir)
        rows = load_csv_rows(summary["seed_aggregated_summary_csv"]) if summary else []

    csv_path = output_path / "seed_summary.csv"
    md_path = output_path / "seed_summary.md"
    tex_path = output_path / "seed_summary.tex"

    if summary_csv:
        csv_path.write_text(Path(summary_csv).read_text(encoding="utf-8"), encoding="utf-8")
    else:
        source_csv = Path(aggregate_dir) / "seed_aggregated_summary.csv"
        if source_csv.exists():
            csv_path.write_text(source_csv.read_text(encoding="utf-8"), encoding="utf-8")

    md_path.write_text(_rows_to_markdown(rows), encoding="utf-8")
    tex_path.write_text(_rows_to_latex(rows), encoding="utf-8")
    return {
        "csv": str(csv_path),
        "markdown": str(md_path),
        "latex": str(tex_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Step 3 paper tables.")
    parser.add_argument("--summary-csv", type=str, default=None)
    parser.add_argument("--output-root", action="append", dest="output_roots", default=None)
    parser.add_argument("--aggregate-dir", type=str, default="vdt_dev/outputs/aggregated")
    parser.add_argument("--output-dir", type=str, default="vdt_dev/paper_tables")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = export_tables(
        summary_csv=args.summary_csv,
        output_roots=args.output_roots,
        aggregate_dir=args.aggregate_dir,
        output_dir=args.output_dir,
    )
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
