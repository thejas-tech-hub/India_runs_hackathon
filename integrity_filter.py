"""
integrity_filter.py
Stage 0A — IntegrityFilter for INDIA.RUNS Data & AI Challenge

Team:    AlgoRhythms
Student: THEJAS J

Detects logical impossibilities (honeypots) in candidate data before
retrieval and ranking.  Operates as a streaming processor over
gzipped (or plain) JSONL files.

Architecture:
    process_stream(filepath) -> Iterator[dict]
        Reads one JSON object per line via gzip.open / open,
        runs all integrity checks, updates internal telemetry,
        and yields a result dict per candidate.

Checks implemented:
    A. Experience vs Education   (FLAG)
    B. Skill Duration            (FLAG)
    C. Future Dates              (FLAG)
    D. Career Consistency        (FLAG)
    E. Missing Fields            (WARN only — never honeypot)
"""

from __future__ import annotations

import gzip
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Iterator, Optional


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class Anomaly:
    """A single integrity anomaly detected for a candidate."""

    candidate_id: str
    check_name: str       # e.g. "experience_vs_education"
    severity: str         # "FLAG" | "WARN"
    message: str          # human-readable explanation
    details: dict         # raw values that triggered the anomaly


@dataclass
class CheckStat:
    """Per-check telemetry counters."""

    candidates_evaluated: int = 0
    candidates_skipped: int = 0
    flags_raised: int = 0
    warnings_raised: int = 0


@dataclass
class TelemetryData:
    """Aggregated telemetry for the entire filter run."""

    run_timestamp: str = ""
    duration_seconds: float = 0.0
    total_candidates: int = 0
    honeypots_detected: int = 0
    clean_passed: int = 0
    warnings_emitted: int = 0
    honeypot_rate: float = 0.0
    warning_rate: float = 0.0
    date_parse_errors: int = 0
    malformed_records: int = 0
    check_stats: dict[str, CheckStat] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

CHECK_NAMES: tuple[str, ...] = (
    "experience_vs_education",
    "skill_duration",
    "future_dates",
    "career_consistency",
    "missing_fields",
)


# ═══════════════════════════════════════════════════════════════════════════
# IntegrityFilter
# ═══════════════════════════════════════════════════════════════════════════

class IntegrityFilter:
    """
    Streaming integrity filter for candidate JSONL data.

    Usage::

        filt = IntegrityFilter()
        for result in filt.process_stream("candidates.jsonl.gz"):
            if result["is_honeypot"]:
                ...  # handle honeypot
        filt.write_telemetry("sample_report.json")
    """

    # ── Constants (from challenge requirements) ──────────────────────────
    CURRENT_YEAR: int = 2026
    CURRENT_DATE: date = date(2026, 6, 9)
    BUFFER_YEARS: int = 2
    BUFFER_MONTHS: int = 12

    # ── Init ─────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._telemetry = TelemetryData(
            run_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            check_stats={name: CheckStat() for name in CHECK_NAMES},
        )
        self._start_time: float = 0.0

    # ── Public API ───────────────────────────────────────────────────────

    def process_stream(self, filepath: str) -> Iterator[dict]:
        """
        Stream candidates from a JSONL (optionally gzipped) file.

        Reads one JSON object per line, runs **all** integrity checks
        (never short-circuits), and yields a result dict per candidate::

            {
                "candidate_id": str,
                "is_honeypot":  bool,
                "anomalies":    list[dict],
            }

        Telemetry is finalised when the generator is **fully consumed**.
        Call :meth:`get_telemetry` or :meth:`write_telemetry` afterward.
        """
        self._start_time = time.perf_counter()

        # Transparently handle .gz and plain text
        opener = gzip.open if filepath.endswith(".gz") else open

        with opener(filepath, "rt", encoding="utf-8") as fh:
            for line_num, raw_line in enumerate(fh, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue

                # ── Parse JSON ───────────────────────────────────────
                try:
                    candidate: dict = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Line %d: JSON parse error: %s", line_num, exc
                    )
                    self._telemetry.malformed_records += 1
                    continue

                candidate_id = candidate.get("candidate_id")
                if not candidate_id:
                    logger.warning(
                        "Line %d: Missing candidate_id — skipping", line_num
                    )
                    self._telemetry.malformed_records += 1
                    continue

                self._telemetry.total_candidates += 1

                # ── Run ALL checks (never short-circuit) ─────────────
                anomalies: list[Anomaly] = []
                anomalies.extend(self._check_experience_vs_education(candidate))
                anomalies.extend(self._check_skill_duration(candidate))
                anomalies.extend(self._check_future_dates(candidate))
                anomalies.extend(self._check_career_consistency(candidate))
                anomalies.extend(self._check_missing_fields(candidate))

                # ── Classify ─────────────────────────────────────────
                flags = [a for a in anomalies if a.severity == "FLAG"]
                warns = [a for a in anomalies if a.severity == "WARN"]
                is_honeypot = len(flags) > 0

                if is_honeypot:
                    self._telemetry.honeypots_detected += 1
                else:
                    self._telemetry.clean_passed += 1

                self._telemetry.warnings_emitted += len(warns)

                yield {
                    "candidate_id": candidate_id,
                    "is_honeypot": is_honeypot,
                    "anomalies": [asdict(a) for a in anomalies],
                }

        # ── Finalise telemetry ───────────────────────────────────────
        self._finalise_telemetry()

    def get_telemetry(self) -> dict:
        """Return telemetry as a plain dict.

        Must be called **after** the generator from
        :meth:`process_stream` is fully consumed.
        """
        tel = self._telemetry
        return {
            "run_timestamp": tel.run_timestamp,
            "duration_seconds": tel.duration_seconds,
            "total_candidates": tel.total_candidates,
            "honeypots_detected": tel.honeypots_detected,
            "clean_passed": tel.clean_passed,
            "warnings_emitted": tel.warnings_emitted,
            "honeypot_rate": tel.honeypot_rate,
            "warning_rate": tel.warning_rate,
            "date_parse_errors": tel.date_parse_errors,
            "malformed_records": tel.malformed_records,
            "check_stats": {
                name: asdict(stat)
                for name, stat in tel.check_stats.items()
            },
        }

    def write_telemetry(self, filepath: str = "sample_report.json") -> None:
        """Serialise telemetry to a JSON file on disk."""
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(self.get_telemetry(), fh, indent=2)
        logger.info("Telemetry written to %s", filepath)

    # ── Private: date parsing ────────────────────────────────────────────

    def _parse_date(self, value: object) -> Optional[date]:
        """Parse an ISO-8601 date string.  Returns ``None`` on failure."""
        if value is None:
            return None
        try:
            return date.fromisoformat(str(value))
        except (ValueError, TypeError):
            self._telemetry.date_parse_errors += 1
            logger.warning("Unparseable date value: %r", value)
            return None

    # ── Private: telemetry finalisation ──────────────────────────────────

    def _finalise_telemetry(self) -> None:
        elapsed = time.perf_counter() - self._start_time
        tel = self._telemetry
        tel.duration_seconds = round(elapsed, 3)
        total = tel.total_candidates
        if total > 0:
            tel.honeypot_rate = round(tel.honeypots_detected / total, 5)
            tel.warning_rate = round(tel.warnings_emitted / total, 5)

        logger.info(
            "IntegrityFilter complete: %d candidates | %d honeypots | "
            "%d clean | %d warnings | %.3fs",
            total,
            tel.honeypots_detected,
            tel.clean_passed,
            tel.warnings_emitted,
            elapsed,
        )

    # ══════════════════════════════════════════════════════════════════════
    # CHECK A — Experience vs Education
    # ══════════════════════════════════════════════════════════════════════

    def _check_experience_vs_education(
        self, candidate: dict
    ) -> list[Anomaly]:
        """
        Flag if claimed years_of_experience exceeds
        (years_since_earliest_graduation + BUFFER_YEARS).
        """
        stat = self._telemetry.check_stats["experience_vs_education"]
        cid: str = candidate["candidate_id"]

        profile: dict = candidate.get("profile") or {}
        yoe = profile.get("years_of_experience")
        education: list = candidate.get("education") or []

        # Need both fields to evaluate
        if yoe is None or not education:
            stat.candidates_skipped += 1
            return []

        # Extract graduation years from parseable end_dates
        grad_years: list[int] = []
        for edu in education:
            parsed = self._parse_date(edu.get("end_date"))
            if parsed is not None:
                grad_years.append(parsed.year)

        if not grad_years:
            stat.candidates_skipped += 1
            return []

        stat.candidates_evaluated += 1

        earliest_grad_year = min(grad_years)
        years_since_grad = self.CURRENT_YEAR - earliest_grad_year
        max_plausible = years_since_grad + self.BUFFER_YEARS

        if yoe > max_plausible:
            stat.flags_raised += 1
            return [
                Anomaly(
                    candidate_id=cid,
                    check_name="experience_vs_education",
                    severity="FLAG",
                    message=(
                        f"Claimed {yoe} years experience but earliest "
                        f"graduation was {earliest_grad_year} "
                        f"({years_since_grad} years ago). "
                        f"Max plausible: {max_plausible}"
                    ),
                    details={
                        "years_of_experience": yoe,
                        "earliest_grad_year": earliest_grad_year,
                        "years_since_grad": years_since_grad,
                        "max_plausible": max_plausible,
                    },
                )
            ]
        return []

    # ══════════════════════════════════════════════════════════════════════
    # CHECK B — Skill Duration
    # ══════════════════════════════════════════════════════════════════════

    def _check_skill_duration(self, candidate: dict) -> list[Anomaly]:
        """
        Flag each skill whose duration_months exceeds
        (years_of_experience × 12 + BUFFER_MONTHS).
        """
        stat = self._telemetry.check_stats["skill_duration"]
        cid: str = candidate["candidate_id"]

        profile: dict = candidate.get("profile") or {}
        yoe = profile.get("years_of_experience")
        skills: list = candidate.get("skills") or []

        if yoe is None or not skills:
            stat.candidates_skipped += 1
            return []

        stat.candidates_evaluated += 1
        max_skill_months = yoe * 12 + self.BUFFER_MONTHS
        anomalies: list[Anomaly] = []

        for skill in skills:
            duration = skill.get("duration_months")
            name = skill.get("name", "<unknown>")

            if duration is None:
                logger.warning(
                    "Candidate %s: skill %r has null duration_months — "
                    "skipping skill",
                    cid,
                    name,
                )
                continue

            if duration > max_skill_months:
                stat.flags_raised += 1
                anomalies.append(
                    Anomaly(
                        candidate_id=cid,
                        check_name="skill_duration",
                        severity="FLAG",
                        message=(
                            f"Skill '{name}' claims {duration} months but "
                            f"max plausible is {max_skill_months} months "
                            f"({yoe} yrs x 12 + {self.BUFFER_MONTHS} buffer)"
                        ),
                        details={
                            "skill_name": name,
                            "claimed_months": duration,
                            "max_plausible_months": max_skill_months,
                            "years_of_experience": yoe,
                        },
                    )
                )

        return anomalies

    # ══════════════════════════════════════════════════════════════════════
    # CHECK C — Future Dates
    # ══════════════════════════════════════════════════════════════════════

    def _check_future_dates(self, candidate: dict) -> list[Anomaly]:
        """
        Flag any education end_date, career start_date, or career
        end_date that falls after CURRENT_DATE (2026-06-09).

        A null career end_date is treated as "currently employed"
        and is NOT flagged.
        """
        stat = self._telemetry.check_stats["future_dates"]
        cid: str = candidate["candidate_id"]
        anomalies: list[Anomaly] = []

        stat.candidates_evaluated += 1

        # ── Education future dates ───────────────────────────────────
        for edu in candidate.get("education") or []:
            end = self._parse_date(edu.get("end_date"))
            if end is not None and end > self.CURRENT_DATE:
                stat.flags_raised += 1
                anomalies.append(
                    Anomaly(
                        candidate_id=cid,
                        check_name="future_dates",
                        severity="FLAG",
                        message=(
                            f"Education end_date {end.isoformat()} is after "
                            f"current date {self.CURRENT_DATE.isoformat()}"
                        ),
                        details={
                            "field": "education.end_date",
                            "value": end.isoformat(),
                            "current_date": self.CURRENT_DATE.isoformat(),
                        },
                    )
                )

        # ── Career history future dates ──────────────────────────────
        for entry in candidate.get("career_history") or []:
            start = self._parse_date(entry.get("start_date"))
            if start is not None and start > self.CURRENT_DATE:
                stat.flags_raised += 1
                anomalies.append(
                    Anomaly(
                        candidate_id=cid,
                        check_name="future_dates",
                        severity="FLAG",
                        message=(
                            f"Career start_date {start.isoformat()} is after "
                            f"current date {self.CURRENT_DATE.isoformat()}"
                        ),
                        details={
                            "field": "career_history.start_date",
                            "value": start.isoformat(),
                            "current_date": self.CURRENT_DATE.isoformat(),
                        },
                    )
                )

            end = self._parse_date(entry.get("end_date"))
            if end is not None and end > self.CURRENT_DATE:
                stat.flags_raised += 1
                anomalies.append(
                    Anomaly(
                        candidate_id=cid,
                        check_name="future_dates",
                        severity="FLAG",
                        message=(
                            f"Career end_date {end.isoformat()} is after "
                            f"current date {self.CURRENT_DATE.isoformat()}"
                        ),
                        details={
                            "field": "career_history.end_date",
                            "value": end.isoformat(),
                            "current_date": self.CURRENT_DATE.isoformat(),
                        },
                    )
                )

        return anomalies

    # ══════════════════════════════════════════════════════════════════════
    # CHECK D — Career Consistency
    # ══════════════════════════════════════════════════════════════════════

    def _check_career_consistency(self, candidate: dict) -> list[Anomaly]:
        """
        Flag career entries where:
        - end_date < start_date  (skip if end_date is null → current role)
        - duration_months < 0
        """
        stat = self._telemetry.check_stats["career_consistency"]
        cid: str = candidate["candidate_id"]
        career: list = candidate.get("career_history") or []

        if not career:
            stat.candidates_skipped += 1
            return []

        stat.candidates_evaluated += 1
        anomalies: list[Anomaly] = []

        for entry in career:
            start = self._parse_date(entry.get("start_date"))
            end = self._parse_date(entry.get("end_date"))

            # end_date before start_date (null end = current role → skip)
            if start is not None and end is not None and end < start:
                stat.flags_raised += 1
                anomalies.append(
                    Anomaly(
                        candidate_id=cid,
                        check_name="career_consistency",
                        severity="FLAG",
                        message=(
                            f"Career entry end_date {end.isoformat()} is "
                            f"before start_date {start.isoformat()}"
                        ),
                        details={
                            "start_date": start.isoformat(),
                            "end_date": end.isoformat(),
                        },
                    )
                )

            # Negative duration
            duration = entry.get("duration_months")
            if duration is not None and duration < 0:
                stat.flags_raised += 1
                anomalies.append(
                    Anomaly(
                        candidate_id=cid,
                        check_name="career_consistency",
                        severity="FLAG",
                        message=(
                            f"Career entry has negative duration: "
                            f"{duration} months"
                        ),
                        details={"duration_months": duration},
                    )
                )

        return anomalies

    # ══════════════════════════════════════════════════════════════════════
    # CHECK E — Missing Fields  (WARN only — never marks honeypot)
    # ══════════════════════════════════════════════════════════════════════

    def _check_missing_fields(self, candidate: dict) -> list[Anomaly]:
        """
        Emit WARN-level anomalies for missing critical fields.
        These are informational — they must NEVER cause a candidate
        to be classified as a honeypot.
        """
        stat = self._telemetry.check_stats["missing_fields"]
        cid: str = candidate["candidate_id"]
        anomalies: list[Anomaly] = []

        stat.candidates_evaluated += 1

        profile: dict = candidate.get("profile") or {}
        if profile.get("years_of_experience") is None:
            stat.warnings_raised += 1
            anomalies.append(
                Anomaly(
                    candidate_id=cid,
                    check_name="missing_fields",
                    severity="WARN",
                    message="Missing profile.years_of_experience",
                    details={"field": "profile.years_of_experience"},
                )
            )

        if not candidate.get("education"):
            stat.warnings_raised += 1
            anomalies.append(
                Anomaly(
                    candidate_id=cid,
                    check_name="missing_fields",
                    severity="WARN",
                    message="Missing or empty education",
                    details={"field": "education"},
                )
            )

        if not candidate.get("skills"):
            stat.warnings_raised += 1
            anomalies.append(
                Anomaly(
                    candidate_id=cid,
                    check_name="missing_fields",
                    severity="WARN",
                    message="Missing or empty skills",
                    details={"field": "skills"},
                )
            )

        return anomalies


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    if len(sys.argv) < 2:
        print(
            "Usage: python integrity_filter.py "
            "<input.jsonl[.gz]> [output_report.json]"
        )
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "sample_report.json"

    filt = IntegrityFilter()
    honeypot_count = 0
    clean_count = 0

    for result in filt.process_stream(input_path):
        if result["is_honeypot"]:
            honeypot_count += 1
        else:
            clean_count += 1

    filt.write_telemetry(output_path)

    print(f"\nDone.  Honeypots: {honeypot_count}  |  Clean: {clean_count}")
    print(f"Report written to: {output_path}")
