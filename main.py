"""
main.py
Master Orchestrator for INDIA.RUNS Data & AI Challenge

Team:    AlgoRhythms
Student: THEJAS J

Purpose:
    Wire all 5 completed stages into a single, executable script
    that ingests a JD and a gzipped JSONL candidate file, and
    produces a comprehensive final_report.json.

Pipeline:
    Pass 1  — Stage 0A: IntegrityFilter  (stream, collect clean IDs)
    Pass 2  — Stage 0B: HireabilityEvaluator  (re-read, evaluate clean)
    Stage 1 — HybridRetriever  (BM25 + Dense + RRF → Top K)
    Stage 2 — FinalRanker      (Cross-Encoder + behavioral fusion)
    Stage 3 — ReasoningEngine  (LLM justifications for Top N)

Usage:
    python main.py <candidates.jsonl.gz> <jd.txt> <model.gguf> \
        [--output final_report.json] [--top-k 500] [--top-n 10]
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

# ── Local stage imports ──────────────────────────────────────────────────
from integrity_filter import IntegrityFilter
from hireability_evaluator import HireabilityEvaluator, HireabilityResult
from hybrid_retriever import HybridRetriever
from final_ranker import FinalRanker, RankedCandidate
from reasoning_engine import ReasoningConfig, ReasoningEngine
from submission_exporter import export_submission_csv

# ── Logging ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# ── Constants ────────────────────────────────────────────────────────────

_TEAM = "AlgoRhythms"
_STUDENT = "THEJAS J"
_CHALLENGE = "INDIA.RUNS Data & AI Challenge"
_TIME_LIMIT_SECONDS = 300  # 5-minute hard constraint


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "INDIA.RUNS Pipeline — Stream candidates through integrity "
            "filtering, hireability evaluation, hybrid retrieval, "
            "cross-encoder re-ranking, and LLM justification."
        ),
    )
    parser.add_argument(
        "candidates_file",
        type=str,
        help="Path to candidates.jsonl.gz (gzipped JSONL).",
    )
    parser.add_argument(
        "jd_file",
        type=str,
        help="Path to the Job Description text file (e.g., jd.txt).",
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the .gguf model file for Stage 3 (ReasoningEngine).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="final_report.json",
        help="Output report path (default: final_report.json).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=500,
        dest="top_k",
        help="Top K candidates for Stage 1 HybridRetriever (default: 500).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        dest="top_n",
        help="Top N candidates for Stage 3 LLM justification (default: 10).",
    )
    parser.add_argument(
        "--submission-csv",
        type=str,
        default="submission.csv",
        dest="submission_csv",
        help="Output submission CSV path (default: submission.csv).",
    )
    return parser


# ═══════════════════════════════════════════════════════════════════════════
# File Validation
# ═══════════════════════════════════════════════════════════════════════════


def _validate_inputs(
    candidates_file: str, jd_file: str, model_path: str
) -> None:
    """Validate that all required input files exist.

    Raises:
        SystemExit: If any file is missing, exits with code 1.
    """
    missing: List[str] = []

    if not Path(candidates_file).is_file():
        missing.append(f"Candidates file not found: {candidates_file}")
    if not Path(jd_file).is_file():
        missing.append(f"JD file not found: {jd_file}")
    if not Path(model_path).is_file():
        missing.append(f"Model file not found: {model_path}")

    if missing:
        for msg in missing:
            logger.error(msg)
        print(
            "\n✗ ERROR: Required input file(s) missing.\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nUsage: python main.py <candidates.jsonl.gz> "
            "<jd.txt> <model.gguf>"
        )
        sys.exit(1)


def _load_jd(jd_file: str) -> str:
    """Load the job description text from a file.

    Returns:
        The full JD text as a string.

    Raises:
        SystemExit: If the file cannot be read.
    """
    try:
        text = Path(jd_file).read_text(encoding="utf-8").strip()
        if not text:
            logger.error("JD file is empty: %s", jd_file)
            print(f"\n✗ ERROR: JD file is empty: {jd_file}")
            sys.exit(1)
        logger.info("Loaded JD from %s (%d chars)", jd_file, len(text))
        return text
    except Exception as exc:
        logger.error("Failed to read JD file: %s — %s", jd_file, exc)
        print(f"\n✗ ERROR: Failed to read JD file: {jd_file}\n  {exc}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# Pass 1 — Stage 0A: IntegrityFilter
# ═══════════════════════════════════════════════════════════════════════════


def _run_pass1_integrity_filter(
    candidates_file: str,
) -> tuple[Set[str], dict]:
    """Stream candidates through IntegrityFilter.

    Returns:
        Tuple of (clean_ids set, telemetry dict).
    """
    logger.info("═══ Pass 1 — Stage 0A: IntegrityFilter ═══")
    t0 = time.perf_counter()

    filt = IntegrityFilter()
    clean_ids: Set[str] = set()
    honeypot_count = 0

    for result in filt.process_stream(candidates_file):
        if result["is_honeypot"]:
            honeypot_count += 1
        else:
            clean_ids.add(result["candidate_id"])

    elapsed = time.perf_counter() - t0

    # IntegrityFilter finalises its own telemetry when the generator
    # is fully consumed — retrieve it now.
    telemetry = filt.get_telemetry()

    logger.info(
        "Stage 0A complete: %d clean | %d honeypots | %.3fs",
        len(clean_ids),
        honeypot_count,
        elapsed,
    )

    return clean_ids, telemetry


# ═══════════════════════════════════════════════════════════════════════════
# Pass 2 — Stage 0B: HireabilityEvaluator (+ data collection)
# ═══════════════════════════════════════════════════════════════════════════


def _run_pass2_hireability(
    candidates_file: str,
    clean_ids: Set[str],
) -> tuple[List[Dict[str, Any]], Dict[str, HireabilityResult], float]:
    """Re-read candidates file, collect clean candidates, evaluate.

    Returns:
        Tuple of (clean_candidates list, hireability_results dict,
        stage duration in seconds).
    """
    logger.info("═══ Pass 2 — Stage 0B: HireabilityEvaluator ═══")
    t0 = time.perf_counter()

    evaluator = HireabilityEvaluator()
    clean_candidates: List[Dict[str, Any]] = []
    hireability_results: Dict[str, HireabilityResult] = {}

    # Re-read the gzipped JSONL — transparently handle .gz and plain
    opener = gzip.open if candidates_file.endswith(".gz") else open

    with opener(candidates_file, "rt", encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue

            try:
                candidate: Dict[str, Any] = json.loads(stripped)
            except json.JSONDecodeError:
                continue  # Already counted as malformed in Pass 1

            cid = candidate.get("candidate_id")
            if cid is None or cid not in clean_ids:
                continue

            # Evaluate hireability
            hr: HireabilityResult = evaluator.evaluate(candidate)
            hireability_results[cid] = hr

            # Store raw candidate for Stage 1
            clean_candidates.append(candidate)

    elapsed = time.perf_counter() - t0

    logger.info(
        "Stage 0B complete: %d candidates evaluated | %.3fs",
        len(clean_candidates),
        elapsed,
    )

    return clean_candidates, hireability_results, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 — HybridRetriever
# ═══════════════════════════════════════════════════════════════════════════


def _run_stage1_retrieval(
    jd_text: str,
    clean_candidates: List[Dict[str, Any]],
    top_k: int,
) -> tuple[List[str], float]:
    """Run HybridRetriever to get Top K candidate IDs.

    Returns:
        Tuple of (top_k_ids list, stage duration in seconds).
    """
    logger.info("═══ Stage 1: HybridRetriever (top_k=%d) ═══", top_k)
    t0 = time.perf_counter()

    retriever = HybridRetriever(top_k=top_k)
    top_ids: List[str] = retriever.retrieve(jd_text, clean_candidates, top_k)

    elapsed = time.perf_counter() - t0

    logger.info(
        "Stage 1 complete: %d candidates returned | %.3fs",
        len(top_ids),
        elapsed,
    )

    return top_ids, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 — FinalRanker
# ═══════════════════════════════════════════════════════════════════════════


def _run_stage2_ranking(
    jd_text: str,
    top_candidates: List[Dict[str, Any]],
    hireability_results: Dict[str, HireabilityResult],
) -> tuple[List[RankedCandidate], float]:
    """Run FinalRanker on the Top K candidates.

    Returns:
        Tuple of (ranked_candidates list, stage duration in seconds).
    """
    logger.info(
        "═══ Stage 2: FinalRanker (%d candidates) ═══",
        len(top_candidates),
    )
    t0 = time.perf_counter()

    ranker = FinalRanker()
    ranked: List[RankedCandidate] = ranker.rank(
        jd_text, top_candidates, hireability_results
    )

    elapsed = time.perf_counter() - t0

    logger.info(
        "Stage 2 complete: Top candidate %s (%.4f) | %.3fs",
        ranked[0].candidate_id if ranked else "N/A",
        ranked[0].final_score if ranked else 0.0,
        elapsed,
    )

    return ranked, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3 — ReasoningEngine
# ═══════════════════════════════════════════════════════════════════════════


def _run_stage3_reasoning(
    jd_text: str,
    top_n_candidates: List[RankedCandidate],
    model_path: str,
) -> tuple[List[dict], float]:
    """Run ReasoningEngine on the Top N candidates.

    Returns:
        Tuple of (reasoning_results as dicts, stage duration in seconds).
    """
    logger.info(
        "═══ Stage 3: ReasoningEngine (%d candidates) ═══",
        len(top_n_candidates),
    )
    t0 = time.perf_counter()

    config = ReasoningConfig(model_path=model_path)
    engine = ReasoningEngine(config)

    results = engine.generate_justifications(jd_text, top_n_candidates)
    results_dicts = [r.to_dict() for r in results]

    elapsed = time.perf_counter() - t0

    logger.info(
        "Stage 3 complete: %d justifications | %.3fs",
        len(results_dicts),
        elapsed,
    )

    return results_dicts, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# Report Assembly
# ═══════════════════════════════════════════════════════════════════════════


def _assemble_report(
    *,
    candidates_file: str,
    jd_file: str,
    model_path: str,
    stage_0a_telemetry: dict,
    stage_0b_duration: float,
    stage_0b_count: int,
    stage_1_duration: float,
    stage_1_input_count: int,
    stage_1_output_count: int,
    stage_2_duration: float,
    stage_2_input_count: int,
    stage_3_duration: float,
    stage_3_count: int,
    reasoning_results: List[dict],
    ranked_candidates: List[RankedCandidate],
    total_duration: float,
) -> Dict[str, Any]:
    """Assemble the final report JSON structure."""

    # ── Top 10 candidates with full justifications ───────────────────
    top_candidates: List[Dict[str, Any]] = []

    # Build a lookup from reasoning results keyed by candidate_id
    justification_lookup: Dict[str, dict] = {
        r["candidate_id"]: r for r in reasoning_results
    }

    for idx, rc in enumerate(ranked_candidates[: len(reasoning_results)]):
        justification = justification_lookup.get(rc.candidate_id, {})
        top_candidates.append(
            {
                "rank": idx + 1,
                "candidate_id": rc.candidate_id,
                "final_score": round(rc.final_score, 4),
                "scores": {
                    "norm_semantic_score": round(
                        rc.norm_semantic_score, 4
                    ),
                    "availability_score": round(rc.availability_score, 4),
                    "evidence_coverage_score": round(
                        rc.evidence_coverage_score, 4
                    ),
                    "risk_score": round(rc.risk_score, 4),
                },
                "components": rc.components,
                "justification": {
                    "why_selected": justification.get(
                        "why_selected", "N/A"
                    ),
                    "risk_factors": justification.get(
                        "risk_factors", "N/A"
                    ),
                    "generation_time_ms": justification.get(
                        "generation_time_ms", 0.0
                    ),
                    "prompt_tokens": justification.get(
                        "prompt_tokens", 0
                    ),
                    "completion_tokens": justification.get(
                        "completion_tokens", 0
                    ),
                },
            }
        )

    # ── Top 100 lightweight ranking ──────────────────────────────────
    top_100_ranking: List[Dict[str, Any]] = [
        {
            "rank": idx + 1,
            "candidate_id": rc.candidate_id,
            "final_score": round(rc.final_score, 4),
        }
        for idx, rc in enumerate(ranked_candidates[:100])
    ]

    # ── Average generation time for Stage 3 ──────────────────────────
    gen_times = [
        r.get("generation_time_ms", 0.0) for r in reasoning_results
    ]
    avg_gen_time = (
        round(sum(gen_times) / len(gen_times), 2)
        if gen_times
        else 0.0
    )

    # ── Assemble ─────────────────────────────────────────────────────
    report: Dict[str, Any] = {
        "metadata": {
            "team": _TEAM,
            "student": _STUDENT,
            "challenge": _CHALLENGE,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "jd_file": jd_file,
            "candidates_file": candidates_file,
            "total_pipeline_duration_seconds": round(total_duration, 3),
        },
        "stage_telemetry": {
            "stage_0a_integrity_filter": stage_0a_telemetry,
            "stage_0b_hireability_evaluator": {
                "duration_seconds": round(stage_0b_duration, 3),
                "candidates_evaluated": stage_0b_count,
            },
            "stage_1_hybrid_retriever": {
                "duration_seconds": round(stage_1_duration, 3),
                "candidates_input": stage_1_input_count,
                "top_k_returned": stage_1_output_count,
            },
            "stage_2_final_ranker": {
                "duration_seconds": round(stage_2_duration, 3),
                "candidates_input": stage_2_input_count,
                "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            },
            "stage_3_reasoning_engine": {
                "duration_seconds": round(stage_3_duration, 3),
                "candidates_justified": stage_3_count,
                "model": Path(model_path).name,
                "avg_generation_time_ms": avg_gen_time,
            },
        },
        "top_candidates": top_candidates,
        "top_100_ranking": top_100_ranking,
    }

    return report


# ═══════════════════════════════════════════════════════════════════════════
# Console Summary
# ═══════════════════════════════════════════════════════════════════════════


def _print_summary(stage_timings: Dict[str, float], total: float) -> None:
    """Print a formatted execution timing summary to stdout."""
    passed = total <= _TIME_LIMIT_SECONDS
    status = "✓ PASS" if passed else "✗ FAIL (exceeded 5 min)"

    print()
    print("═══════════════════════════════════════════════════════")
    print("  INDIA.RUNS Pipeline — Execution Summary")
    print("═══════════════════════════════════════════════════════")
    print(
        f"  Stage 0A  IntegrityFilter ........"
        f"  {stage_timings['stage_0a']:>7.3f}s"
    )
    print(
        f"  Stage 0B  HireabilityEvaluator ..."
        f"  {stage_timings['stage_0b']:>7.3f}s"
    )
    print(
        f"  Stage 1   HybridRetriever ........"
        f"  {stage_timings['stage_1']:>7.3f}s"
    )
    print(
        f"  Stage 2   FinalRanker ............"
        f"  {stage_timings['stage_2']:>7.3f}s"
    )
    print(
        f"  Stage 3   ReasoningEngine ........"
        f"  {stage_timings['stage_3']:>7.3f}s"
    )
    print("  ─────────────────────────────────────────────")
    print(f"  TOTAL ............................  {total:>7.3f}s  {status}")
    print("═══════════════════════════════════════════════════════")


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline Orchestration
# ═══════════════════════════════════════════════════════════════════════════


def run_pipeline(
    candidates_file: str,
    jd_file: str,
    model_path: str,
    output_file: str,
    top_k: int,
    top_n: int,
    submission_csv: str = "submission.csv",
) -> None:
    """Execute the full 5-stage pipeline end-to-end.

    Args:
        candidates_file: Path to candidates.jsonl.gz.
        jd_file: Path to jd.txt.
        model_path: Path to .gguf model for Stage 3.
        output_file: Path to write final_report.json.
        top_k: Number of candidates for Stage 1 retrieval.
        top_n: Number of candidates for Stage 3 justification.
        submission_csv: Path to write the official submission CSV.
    """
    pipeline_start = time.perf_counter()
    stage_timings: Dict[str, float] = {}

    logger.info(
        "Pipeline starting — candidates=%s  jd=%s  model=%s  "
        "top_k=%d  top_n=%d",
        candidates_file,
        jd_file,
        model_path,
        top_k,
        top_n,
    )

    # ── Load JD ──────────────────────────────────────────────────────
    jd_text: str = _load_jd(jd_file)

    # ══════════════════════════════════════════════════════════════════
    # PASS 1 — Stage 0A: IntegrityFilter
    # ══════════════════════════════════════════════════════════════════

    t0_0a = time.perf_counter()
    clean_ids, stage_0a_telemetry = _run_pass1_integrity_filter(
        candidates_file
    )
    stage_timings["stage_0a"] = time.perf_counter() - t0_0a

    if not clean_ids:
        logger.error(
            "0 clean candidates after Stage 0A — cannot continue."
        )
        # Write partial report with telemetry only
        partial_report: Dict[str, Any] = {
            "metadata": {
                "team": _TEAM,
                "student": _STUDENT,
                "challenge": _CHALLENGE,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "jd_file": jd_file,
                "candidates_file": candidates_file,
                "total_pipeline_duration_seconds": round(
                    time.perf_counter() - pipeline_start, 3
                ),
                "error": "0 clean candidates after Stage 0A",
            },
            "stage_telemetry": {
                "stage_0a_integrity_filter": stage_0a_telemetry,
            },
            "top_candidates": [],
            "top_100_ranking": [],
        }
        with open(output_file, "w", encoding="utf-8") as fh:
            json.dump(partial_report, fh, indent=2)
        logger.info("Partial report written to %s", output_file)
        print(
            f"\n✗ Pipeline aborted: 0 clean candidates. "
            f"Partial report saved to {output_file}"
        )
        sys.exit(2)

    # ══════════════════════════════════════════════════════════════════
    # PASS 2 — Stage 0B: HireabilityEvaluator
    # ══════════════════════════════════════════════════════════════════

    clean_candidates, hireability_results, stage_0b_duration = (
        _run_pass2_hireability(candidates_file, clean_ids)
    )
    stage_timings["stage_0b"] = stage_0b_duration

    # Free the clean_ids set — no longer needed
    n_clean = len(clean_candidates)
    del clean_ids
    gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # STAGE 1 — HybridRetriever
    # ══════════════════════════════════════════════════════════════════

    top_ids, stage_1_duration = _run_stage1_retrieval(
        jd_text, clean_candidates, top_k
    )
    stage_timings["stage_1"] = stage_1_duration

    # ── MEMORY PRUNING: Build Top K subset, discard full list ────────
    logger.info("Memory pruning: extracting Top %d subset...", len(top_ids))

    # Build a fast lookup for the Top K IDs
    top_id_set: Set[str] = set(top_ids)

    # Build ordered candidate lookup preserving Stage 1 ranking order
    candidate_lookup: Dict[str, Dict[str, Any]] = {
        c["candidate_id"]: c
        for c in clean_candidates
        if c.get("candidate_id") in top_id_set
    }

    # Preserve Stage 1 ranking order
    top_candidates: List[Dict[str, Any]] = [
        candidate_lookup[cid] for cid in top_ids if cid in candidate_lookup
    ]

    # Free the massive clean_candidates list
    del clean_candidates
    del candidate_lookup
    del top_id_set
    gc.collect()

    logger.info(
        "Memory pruning complete: %d candidates retained, "
        "full list freed.",
        len(top_candidates),
    )

    # ══════════════════════════════════════════════════════════════════
    # STAGE 2 — FinalRanker
    # ══════════════════════════════════════════════════════════════════

    ranked_candidates, stage_2_duration = _run_stage2_ranking(
        jd_text, top_candidates, hireability_results
    )
    stage_timings["stage_2"] = stage_2_duration

    # ── MEMORY PRUNING: Free Stage 2 inputs ──────────────────────────
    stage_2_input_count = len(top_candidates)
    del top_candidates
    del hireability_results
    gc.collect()

    logger.info(
        "Memory pruning: Stage 2 inputs freed. "
        "%d ranked candidates held.",
        len(ranked_candidates),
    )

    # ══════════════════════════════════════════════════════════════════
    # STAGE 3 — ReasoningEngine
    # ══════════════════════════════════════════════════════════════════

    # Slice only the Top N for LLM justification
    effective_top_n = min(top_n, len(ranked_candidates))
    top_n_for_reasoning: List[RankedCandidate] = (
        ranked_candidates[:effective_top_n]
    )

    reasoning_results, stage_3_duration = _run_stage3_reasoning(
        jd_text, top_n_for_reasoning, model_path
    )
    stage_timings["stage_3"] = stage_3_duration

    # ══════════════════════════════════════════════════════════════════
    # REPORT ASSEMBLY
    # ══════════════════════════════════════════════════════════════════

    total_duration = time.perf_counter() - pipeline_start

    report = _assemble_report(
        candidates_file=candidates_file,
        jd_file=jd_file,
        model_path=model_path,
        stage_0a_telemetry=stage_0a_telemetry,
        stage_0b_duration=stage_0b_duration,
        stage_0b_count=n_clean,
        stage_1_duration=stage_1_duration,
        stage_1_input_count=n_clean,
        stage_1_output_count=len(top_ids),
        stage_2_duration=stage_2_duration,
        stage_2_input_count=stage_2_input_count,
        stage_3_duration=stage_3_duration,
        stage_3_count=effective_top_n,
        reasoning_results=reasoning_results,
        ranked_candidates=ranked_candidates,
        total_duration=total_duration,
    )

    # ── Write report ─────────────────────────────────────────────────
    with open(output_file, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    logger.info("Report written to %s", output_file)

    # ── Generate official submission CSV ─────────────────────────────
    if submission_csv:
        logger.info("Generating submission CSV: %s", submission_csv)
        csv_result = export_submission_csv(output_file, submission_csv)
        if csv_result["valid"]:
            logger.info(
                "Submission CSV validated: %d rows, all checks passed.",
                csv_result["rows"],
            )
        else:
            logger.warning(
                "Submission CSV validation issues: %s",
                csv_result["issues"],
            )

    # ── Console summary ──────────────────────────────────────────────
    _print_summary(stage_timings, total_duration)
    print(f"  Report saved to: {output_file}")
    if submission_csv:
        print(f"  Submission CSV: {submission_csv}")
    print("═══════════════════════════════════════════════════════")


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        _validate_inputs(args.candidates_file, args.jd_file, args.model_path)
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("Input validation failed: %s", exc)
        print(f"\n✗ ERROR: Input validation failed: {exc}")
        sys.exit(1)

    try:
        run_pipeline(
            candidates_file=args.candidates_file,
            jd_file=args.jd_file,
            model_path=args.model_path,
            output_file=args.output,
            top_k=args.top_k,
            top_n=args.top_n,
            submission_csv=args.submission_csv,
        )
    except KeyboardInterrupt:
        print("\n\n✗ Pipeline interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.exception("Pipeline failed with unhandled exception")
        print(f"\n✗ FATAL: Pipeline failed — {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
