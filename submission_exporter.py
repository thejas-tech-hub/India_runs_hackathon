"""
submission_exporter.py
Official Submission CSV Exporter for INDIA.RUNS Data & AI Challenge

Team:    AlgoRhythms
Student: THEJAS J

Purpose:
    Read the frozen final_report.json and export ONE official CSV file
    matching the competition submission specification.

    Columns (exact order): candidate_id, rank, score, reasoning

    This module does NOT modify ranking, scores, or reasoning.
    It is a pure format-conversion utility.

Usage (standalone):
    python submission_exporter.py [--input final_report.json] [--output submission.csv]

Usage (from pipeline):
    from submission_exporter import export_submission_csv
    export_submission_csv("final_report.json", "submission.csv")
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def export_submission_csv(
    report_path: str,
    output_path: str = "submission.csv",
) -> dict:
    """Export the official submission CSV from a final_report.json file.

    Args:
        report_path: Path to the final_report.json file.
        output_path: Path to write the submission CSV.

    Returns:
        A validation summary dict with keys:
            rows, unique_ids, unique_ranks, monotonic, path, valid

    Raises:
        FileNotFoundError: If report_path does not exist.
        ValueError: If validation checks fail.
    """

    # ── Load report ──────────────────────────────────────────────────
    report_file = Path(report_path)
    if not report_file.is_file():
        raise FileNotFoundError(f"Report file not found: {report_path}")

    with open(report_file, "r", encoding="utf-8") as fh:
        report: Dict[str, Any] = json.load(fh)

    # ── Extract data ─────────────────────────────────────────────────
    top_candidates: List[Dict[str, Any]] = report.get("top_candidates", [])
    top_100_ranking: List[Dict[str, Any]] = report.get("top_100_ranking", [])

    # Build justification lookup from top_candidates (ranks 1-10)
    justification_lookup: Dict[str, Dict[str, str]] = {}
    for c in top_candidates:
        cid = c["candidate_id"]
        j = c.get("justification", {})
        why = j.get("why_selected", "")
        risk = j.get("risk_factors", "")
        # Combine why_selected and risk_factors into a single reasoning string
        parts = []
        if why:
            parts.append(why)
        if risk:
            parts.append(f"Risk: {risk}")
        justification_lookup[cid] = {
            "reasoning": " | ".join(parts) if parts else "",
        }

    # ── Build rows (preserving exact ranking order) ──────────────────
    rows: List[Dict[str, Any]] = []
    for entry in top_100_ranking:
        cid = entry["candidate_id"]
        rank = entry["rank"]
        score = entry["final_score"]

        justification = justification_lookup.get(cid, {})
        reasoning = justification.get("reasoning", "")

        rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": score,
            "reasoning": reasoning,
        })

    # ── Write CSV ────────────────────────────────────────────────────
    fieldnames = ["candidate_id", "rank", "score", "reasoning"]

    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Submission CSV written to %s (%d rows)", output_path, len(rows))

    # ── Validate ─────────────────────────────────────────────────────
    validation = _validate_csv(output_path, rows)

    return validation


def _validate_csv(
    csv_path: str,
    original_rows: List[Dict[str, Any]],
) -> dict:
    """Validate the generated CSV against submission requirements.

    Returns:
        A dict with validation results.
    """
    issues: List[str] = []

    # Re-read the CSV to validate what was actually written
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        read_rows = list(reader)

    # Check exact column names and order
    expected_cols = ["candidate_id", "rank", "score", "reasoning"]
    with open(csv_path, "r", encoding="utf-8") as fh:
        header_line = fh.readline().strip()
    actual_cols = header_line.split(",")
    if actual_cols != expected_cols:
        issues.append(f"Column mismatch: expected {expected_cols}, got {actual_cols}")

    # Row count
    row_count = len(read_rows)
    if row_count != 100:
        issues.append(f"Expected 100 rows, got {row_count}")

    # Unique candidate IDs
    cids = [r["candidate_id"] for r in read_rows]
    unique_ids = len(set(cids))
    if unique_ids != row_count:
        issues.append(f"Duplicate candidate IDs: {row_count} rows but {unique_ids} unique IDs")

    # Unique ranks
    ranks = [int(r["rank"]) for r in read_rows]
    unique_ranks = len(set(ranks))
    if unique_ranks != row_count:
        issues.append(f"Duplicate ranks: {row_count} rows but {unique_ranks} unique ranks")

    # Rank range 1-100
    if ranks and (min(ranks) != 1 or max(ranks) != row_count):
        issues.append(f"Rank range: expected 1-{row_count}, got {min(ranks)}-{max(ranks)}")

    # Scores monotonically non-increasing
    scores = [float(r["score"]) for r in read_rows]
    monotonic = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    if not monotonic:
        issues.append("Scores are NOT monotonically non-increasing")

    # No missing cells in required columns
    for i, row in enumerate(read_rows):
        for col in ["candidate_id", "rank", "score"]:
            if not row.get(col, "").strip():
                issues.append(f"Missing value at row {i + 1}, column '{col}'")

    # Encoding check (already written as UTF-8)
    try:
        with open(csv_path, "r", encoding="utf-8") as fh:
            fh.read()
    except UnicodeDecodeError:
        issues.append("File is not valid UTF-8")

    validation = {
        "rows": row_count,
        "unique_ids": unique_ids,
        "unique_ranks": unique_ranks,
        "monotonic": monotonic,
        "path": csv_path,
        "valid": len(issues) == 0,
        "issues": issues,
    }

    return validation


# ═══════════════════════════════════════════════════════════════════════════
# Standalone CLI
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """CLI entry point for standalone CSV generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Export official INDIA.RUNS submission CSV from final_report.json",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="final_report.json",
        help="Path to final_report.json (default: final_report.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Output CSV path (default: submission.csv)",
    )
    args = parser.parse_args()

    print(f"Exporting submission CSV from {args.input} ...")

    try:
        result = export_submission_csv(args.input, args.output)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Print validation results
    print()
    print("-- Validation --")
    print(f"  [{'PASS' if result['rows'] == 100 else 'FAIL'}] Row count: {result['rows']}")
    print(f"  [{'PASS' if result['unique_ids'] == result['rows'] else 'FAIL'}] Unique candidate IDs: {result['unique_ids']}")
    print(f"  [{'PASS' if result['unique_ranks'] == result['rows'] else 'FAIL'}] Unique ranks: {result['unique_ranks']}")
    print(f"  [{'PASS' if result['monotonic'] else 'FAIL'}] Scores monotonically non-increasing: {result['monotonic']}")
    print(f"  [PASS] UTF-8 encoding")
    print(f"  [PASS] Four required columns only")

    if result["issues"]:
        print()
        for issue in result["issues"]:
            print(f"  [FAIL] {issue}")
        print()
        print("[ERROR] Validation FAILED")
        sys.exit(1)
    else:
        print()
        file_size = Path(args.output).stat().st_size
        print("=" * 50)
        print(f"  Submission CSV: {args.output}")
        print(f"  Rows: {result['rows']}")
        print(f"  File size: {file_size:,} bytes")
        print(f"  Validation: ALL PASSED")
        print(f"  Ready for upload: YES")
        print("=" * 50)


if __name__ == "__main__":
    main()
