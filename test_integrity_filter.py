"""
test_integrity_filter.py
Unit tests for Stage 0A IntegrityFilter

Team:    AlgoRhythms
Student: THEJAS J

Run:  pytest test_integrity_filter.py -v
"""

from __future__ import annotations

import gzip
import json
import os
import tempfile

import pytest

from integrity_filter import IntegrityFilter


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_candidate(**overrides) -> dict:
    """Build a baseline *clean* candidate that passes all checks."""
    base = {
        "candidate_id": "CAND_TEST_001",
        "profile": {"years_of_experience": 6.9},
        "career_history": [
            {
                "start_date": "2024-03-08",
                "end_date": "2026-05-01",
                "duration_months": 27,
            }
        ],
        "education": [{"end_date": "2018-05-01"}],
        "skills": [{"name": "Python", "duration_months": 48}],
    }
    base.update(overrides)
    return base


def _write_jsonl_gz(candidates: list[dict], path: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for c in candidates:
            fh.write(json.dumps(c) + "\n")


def _write_jsonl(candidates: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for c in candidates:
            fh.write(json.dumps(c) + "\n")


def _run_filter(
    candidates: list[dict], *, use_gz: bool = True
) -> tuple[list[dict], dict]:
    """Write candidates to a temp JSONL file, run IntegrityFilter,
    return ``(results_list, telemetry_dict)``."""
    suffix = ".jsonl.gz" if use_gz else ".jsonl"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if use_gz:
            _write_jsonl_gz(candidates, tmp_path)
        else:
            _write_jsonl(candidates, tmp_path)

        filt = IntegrityFilter()
        results = list(filt.process_stream(tmp_path))
        telemetry = filt.get_telemetry()
        return results, telemetry
    finally:
        os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Clean candidate baseline
# ═══════════════════════════════════════════════════════════════════════════


class TestCleanCandidate:
    """The schema-example candidate must pass all checks cleanly."""

    def test_clean_candidate_not_honeypot(self):
        results, _ = _run_filter([_make_candidate()])
        assert len(results) == 1
        assert results[0]["is_honeypot"] is False

    def test_clean_candidate_zero_anomalies(self):
        results, _ = _run_filter([_make_candidate()])
        assert results[0]["anomalies"] == []

    def test_clean_candidate_telemetry(self):
        _, tel = _run_filter([_make_candidate()])
        assert tel["honeypots_detected"] == 0
        assert tel["clean_passed"] == 1
        assert tel["warnings_emitted"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Check A — Experience vs Education
# ═══════════════════════════════════════════════════════════════════════════


class TestCheckA:
    """Experience exceeding (years_since_grad + BUFFER_YEARS) → FLAG."""

    def test_experience_exceeds_graduation(self):
        # Grad 2018 → years_since = 8 → max = 10.  15 > 10 → FLAG
        cand = _make_candidate()
        cand["profile"]["years_of_experience"] = 15
        results, _ = _run_filter([cand])
        assert results[0]["is_honeypot"] is True
        flags = [
            a
            for a in results[0]["anomalies"]
            if a["check_name"] == "experience_vs_education"
            and a["severity"] == "FLAG"
        ]
        assert len(flags) == 1

    def test_experience_within_buffer_passes(self):
        # max = 10.  10 ≤ 10 → PASS
        cand = _make_candidate()
        cand["profile"]["years_of_experience"] = 10
        results, _ = _run_filter([cand])
        a_flags = [
            a
            for a in results[0]["anomalies"]
            if a["check_name"] == "experience_vs_education"
        ]
        assert len(a_flags) == 0

    def test_experience_exactly_at_boundary(self):
        # 10.0 ≤ 10 → PASS (not strictly greater)
        cand = _make_candidate()
        cand["profile"]["years_of_experience"] = 10.0
        results, _ = _run_filter([cand])
        a_flags = [
            a
            for a in results[0]["anomalies"]
            if a["check_name"] == "experience_vs_education"
        ]
        assert len(a_flags) == 0

    def test_multiple_education_uses_earliest(self):
        # Two degrees: 2016, 2020.  min = 2016 → years_since = 10 → max = 12
        cand = _make_candidate()
        cand["education"] = [
            {"end_date": "2020-05-01"},
            {"end_date": "2016-06-01"},
        ]
        cand["profile"]["years_of_experience"] = 11
        results, _ = _run_filter([cand])
        assert results[0]["is_honeypot"] is False

    def test_skipped_when_no_education(self):
        cand = _make_candidate(education=[])
        _, tel = _run_filter([cand])
        stat = tel["check_stats"]["experience_vs_education"]
        assert stat["candidates_skipped"] == 1
        assert stat["candidates_evaluated"] == 0

    def test_skipped_when_no_yoe(self):
        cand = _make_candidate()
        del cand["profile"]["years_of_experience"]
        _, tel = _run_filter([cand])
        stat = tel["check_stats"]["experience_vs_education"]
        assert stat["candidates_skipped"] == 1

    def test_unparseable_education_dates_skipped(self):
        cand = _make_candidate()
        cand["education"] = [{"end_date": "not-a-date"}]
        _, tel = _run_filter([cand])
        assert tel["date_parse_errors"] >= 1
        stat = tel["check_stats"]["experience_vs_education"]
        assert stat["candidates_skipped"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Check B — Skill Duration
# ═══════════════════════════════════════════════════════════════════════════


class TestCheckB:
    """Skill duration exceeding (yoe × 12 + BUFFER_MONTHS) → FLAG."""

    def test_skill_duration_exceeded(self):
        # 6.9 * 12 + 12 = 94.8.  120 > 94.8 → FLAG
        cand = _make_candidate()
        cand["skills"] = [{"name": "Python", "duration_months": 120}]
        results, _ = _run_filter([cand])
        assert results[0]["is_honeypot"] is True
        flags = [
            a
            for a in results[0]["anomalies"]
            if a["check_name"] == "skill_duration"
        ]
        assert len(flags) == 1

    def test_skill_duration_within_buffer(self):
        # 94 ≤ 94.8 → PASS
        cand = _make_candidate()
        cand["skills"] = [{"name": "Python", "duration_months": 94}]
        results, _ = _run_filter([cand])
        b_flags = [
            a
            for a in results[0]["anomalies"]
            if a["check_name"] == "skill_duration" and a["severity"] == "FLAG"
        ]
        assert len(b_flags) == 0

    def test_multiple_skills_each_checked(self):
        # Two over-limit skills → 2 separate FLAGs
        cand = _make_candidate()
        cand["skills"] = [
            {"name": "Python", "duration_months": 200},
            {"name": "Java", "duration_months": 300},
        ]
        results, _ = _run_filter([cand])
        flags = [
            a
            for a in results[0]["anomalies"]
            if a["check_name"] == "skill_duration"
        ]
        assert len(flags) == 2

    def test_null_duration_skipped_no_flag(self):
        cand = _make_candidate()
        cand["skills"] = [{"name": "Python", "duration_months": None}]
        results, _ = _run_filter([cand])
        flags = [
            a
            for a in results[0]["anomalies"]
            if a["check_name"] == "skill_duration"
        ]
        assert len(flags) == 0

    def test_zero_duration_not_flagged(self):
        cand = _make_candidate()
        cand["skills"] = [{"name": "Python", "duration_months": 0}]
        results, _ = _run_filter([cand])
        flags = [
            a
            for a in results[0]["anomalies"]
            if a["check_name"] == "skill_duration"
        ]
        assert len(flags) == 0

    def test_skipped_when_no_skills(self):
        cand = _make_candidate(skills=[])
        _, tel = _run_filter([cand])
        stat = tel["check_stats"]["skill_duration"]
        assert stat["candidates_skipped"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Check C — Future Dates
# ═══════════════════════════════════════════════════════════════════════════


class TestCheckC:
    """Any date after 2026-06-09 → FLAG."""

    def test_future_education_end_date(self):
        cand = _make_candidate()
        cand["education"] = [{"end_date": "2027-06-01"}]
        results, _ = _run_filter([cand])
        assert results[0]["is_honeypot"] is True
        flags = [
            a
            for a in results[0]["anomalies"]
            if a["check_name"] == "future_dates"
        ]
        assert any(
            a["details"].get("field") == "education.end_date" for a in flags
        )

    def test_future_career_start_date(self):
        cand = _make_candidate()
        cand["career_history"] = [
            {
                "start_date": "2027-01-01",
                "end_date": "2028-01-01",
                "duration_months": 12,
            }
        ]
        results, _ = _run_filter([cand])
        assert results[0]["is_honeypot"] is True
        flags = [
            a
            for a in results[0]["anomalies"]
            if a["check_name"] == "future_dates"
        ]
        assert any(
            a["details"].get("field") == "career_history.start_date"
            for a in flags
        )

    def test_future_career_end_date(self):
        cand = _make_candidate()
        cand["career_history"] = [
            {
                "start_date": "2025-01-01",
                "end_date": "2026-12-01",
                "duration_months": 23,
            }
        ]
        results, _ = _run_filter([cand])
        assert results[0]["is_honeypot"] is True
        flags = [
            a
            for a in results[0]["anomalies"]
            if a["check_name"] == "future_dates"
        ]
        assert any(
            a["details"].get("field") == "career_history.end_date"
            for a in flags
        )

    def test_null_career_end_date_not_flagged(self):
        """null end_date = currently employed → NOT a future date."""
        cand = _make_candidate()
        cand["career_history"] = [
            {
                "start_date": "2025-01-01",
                "end_date": None,
                "duration_months": 17,
            }
        ]
        results, _ = _run_filter([cand])
        future_flags = [
            a
            for a in results[0]["anomalies"]
            if a["check_name"] == "future_dates"
        ]
        assert len(future_flags) == 0

    def test_past_dates_pass(self):
        results, _ = _run_filter([_make_candidate()])
        future_flags = [
            a
            for a in results[0]["anomalies"]
            if a["check_name"] == "future_dates"
        ]
        assert len(future_flags) == 0

    def test_date_exactly_on_current_date_passes(self):
        """2026-06-09 is NOT after 2026-06-09 → PASS."""
        cand = _make_candidate()
        cand["career_history"] = [
            {
                "start_date": "2025-01-01",
                "end_date": "2026-06-09",
                "duration_months": 17,
            }
        ]
        results, _ = _run_filter([cand])
        future_flags = [
            a
            for a in results[0]["anomalies"]
            if a["check_name"] == "future_dates"
        ]
        assert len(future_flags) == 0

    def test_date_one_day_after_current_flagged(self):
        """2026-06-10 IS after 2026-06-09 → FLAG."""
        cand = _make_candidate()
        cand["career_history"] = [
            {
                "start_date": "2025-01-01",
                "end_date": "2026-06-10",
                "duration_months": 17,
            }
        ]
        results, _ = _run_filter([cand])
        future_flags = [
            a
            for a in results[0]["anomalies"]
            if a["check_name"] == "future_dates"
        ]
        assert len(future_flags) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Check D — Career Consistency
# ═══════════════════════════════════════════════════════════════════════════


class TestCheckD:
    """end_date < start_date or duration_months < 0 → FLAG."""

    def test_end_before_start(self):
        cand = _make_candidate()
        cand["career_history"] = [
            {
                "start_date": "2024-06-01",
                "end_date": "2023-01-01",
                "duration_months": 10,
            }
        ]
        results, _ = _run_filter([cand])
        assert results[0]["is_honeypot"] is True
        flags = [
            a
            for a in results[0]["anomalies"]
            if a["check_name"] == "career_consistency"
        ]
        assert len(flags) == 1

    def test_negative_duration(self):
        cand = _make_candidate()
        cand["career_history"] = [
            {
                "start_date": "2024-01-01",
                "end_date": "2025-01-01",
                "duration_months": -5,
            }
        ]
        results, _ = _run_filter([cand])
        assert results[0]["is_honeypot"] is True
        flags = [
            a
            for a in results[0]["anomalies"]
            if a["check_name"] == "career_consistency"
        ]
        assert len(flags) == 1

    def test_null_end_date_no_consistency_flag(self):
        """null end_date → currently employed → skip date-order check."""
        cand = _make_candidate()
        cand["career_history"] = [
            {
                "start_date": "2024-01-01",
                "end_date": None,
                "duration_months": 17,
            }
        ]
        results, _ = _run_filter([cand])
        d_flags = [
            a
            for a in results[0]["anomalies"]
            if a["check_name"] == "career_consistency"
        ]
        assert len(d_flags) == 0

    def test_both_end_before_start_and_negative_duration(self):
        """Both sub-checks trigger → 2 separate anomalies."""
        cand = _make_candidate()
        cand["career_history"] = [
            {
                "start_date": "2025-01-01",
                "end_date": "2024-01-01",
                "duration_months": -12,
            }
        ]
        results, _ = _run_filter([cand])
        flags = [
            a
            for a in results[0]["anomalies"]
            if a["check_name"] == "career_consistency"
        ]
        assert len(flags) == 2

    def test_zero_duration_not_flagged(self):
        cand = _make_candidate()
        cand["career_history"] = [
            {
                "start_date": "2025-01-01",
                "end_date": "2025-01-01",
                "duration_months": 0,
            }
        ]
        results, _ = _run_filter([cand])
        d_flags = [
            a
            for a in results[0]["anomalies"]
            if a["check_name"] == "career_consistency"
        ]
        assert len(d_flags) == 0

    def test_no_career_history_skipped(self):
        cand = _make_candidate()
        cand["career_history"] = []
        _, tel = _run_filter([cand])
        stat = tel["check_stats"]["career_consistency"]
        assert stat["candidates_skipped"] == 1
        assert stat["candidates_evaluated"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Check E — Missing Fields (WARN only)
# ═══════════════════════════════════════════════════════════════════════════


class TestCheckE:
    """Missing fields emit WARNs.  WARNs NEVER mark honeypot."""

    def test_missing_yoe_warns_not_honeypot(self):
        cand = _make_candidate()
        del cand["profile"]["years_of_experience"]
        results, _ = _run_filter([cand])
        assert results[0]["is_honeypot"] is False
        warns = [a for a in results[0]["anomalies"] if a["severity"] == "WARN"]
        assert any("years_of_experience" in a["message"] for a in warns)

    def test_missing_education_warns(self):
        cand = _make_candidate(education=[])
        results, _ = _run_filter([cand])
        assert results[0]["is_honeypot"] is False
        warns = [a for a in results[0]["anomalies"] if a["severity"] == "WARN"]
        assert any("education" in a["message"] for a in warns)

    def test_missing_skills_warns(self):
        cand = _make_candidate(skills=[])
        results, _ = _run_filter([cand])
        assert results[0]["is_honeypot"] is False
        warns = [a for a in results[0]["anomalies"] if a["severity"] == "WARN"]
        assert any("skills" in a["message"] for a in warns)

    def test_missing_profile_entirely(self):
        cand = _make_candidate()
        del cand["profile"]
        results, _ = _run_filter([cand])
        assert results[0]["is_honeypot"] is False
        warns = [a for a in results[0]["anomalies"] if a["severity"] == "WARN"]
        assert any("years_of_experience" in a["message"] for a in warns)

    def test_all_fields_missing_three_warnings(self):
        cand = {"candidate_id": "CAND_EMPTY"}
        results, _ = _run_filter([cand])
        assert results[0]["is_honeypot"] is False
        warns = [a for a in results[0]["anomalies"] if a["severity"] == "WARN"]
        assert len(warns) == 3

    def test_null_education_warns(self):
        cand = _make_candidate(education=None)
        results, _ = _run_filter([cand])
        warns = [a for a in results[0]["anomalies"] if a["severity"] == "WARN"]
        assert any("education" in a["message"] for a in warns)


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Multi-flag candidates (all checks run, no short-circuit)
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiFlag:
    def test_candidate_tripping_three_checks(self):
        """A, B, and C all flag → all 3+ anomalies appear in output."""
        cand = {
            "candidate_id": "CAND_MULTI",
            "profile": {"years_of_experience": 25},   # A: 25 > 10 → FLAG
            "education": [{"end_date": "2018-05-01"}],
            "skills": [
                {"name": "Python", "duration_months": 500},  # B: 500 > 312 → FLAG
            ],
            "career_history": [
                {
                    "start_date": "2027-01-01",  # C: future → FLAG
                    "end_date": "2028-01-01",    # C: future → FLAG
                    "duration_months": 12,
                }
            ],
        }
        results, _ = _run_filter([cand])
        assert results[0]["is_honeypot"] is True
        flags = [a for a in results[0]["anomalies"] if a["severity"] == "FLAG"]
        check_names = {a["check_name"] for a in flags}
        assert "experience_vs_education" in check_names
        assert "skill_duration" in check_names
        assert "future_dates" in check_names
        # Must have at least 4 flags (A=1, B=1, C=2)
        assert len(flags) >= 4

    def test_candidate_with_all_five_checks_triggered(self):
        """A, B, C, D flags + E warns.  All anomalies reported."""
        cand = {
            "candidate_id": "CAND_ALL",
            "profile": {"years_of_experience": 50},
            "education": [{"end_date": "2028-01-01"}],
            "skills": [{"name": "Rust", "duration_months": 9999}],
            "career_history": [
                {
                    "start_date": "2029-01-01",
                    "end_date": "2020-01-01",
                    "duration_months": -36,
                }
            ],
        }
        results, _ = _run_filter([cand])
        assert results[0]["is_honeypot"] is True
        all_checks = {a["check_name"] for a in results[0]["anomalies"]}
        assert "experience_vs_education" in all_checks
        assert "skill_duration" in all_checks
        assert "future_dates" in all_checks
        assert "career_consistency" in all_checks
        # Check E should NOT contribute flags (WARN only)
        # but it should not exist here since fields are present


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Streaming & file handling
# ═══════════════════════════════════════════════════════════════════════════


class TestStreaming:
    def test_gzipped_input(self):
        results, _ = _run_filter([_make_candidate()], use_gz=True)
        assert len(results) == 1

    def test_plain_jsonl_input(self):
        results, _ = _run_filter([_make_candidate()], use_gz=False)
        assert len(results) == 1

    def test_empty_file(self):
        results, tel = _run_filter([])
        assert results == []
        assert tel["total_candidates"] == 0

    def test_malformed_json_skipped(self):
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", mode="w", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write("NOT VALID JSON\n")
            tmp.write(json.dumps(_make_candidate()) + "\n")
            tmp_path = tmp.name

        try:
            filt = IntegrityFilter()
            results = list(filt.process_stream(tmp_path))
            tel = filt.get_telemetry()
            assert len(results) == 1
            assert tel["malformed_records"] == 1
        finally:
            os.unlink(tmp_path)

    def test_missing_candidate_id_skipped(self):
        cand = _make_candidate()
        del cand["candidate_id"]
        results, tel = _run_filter([cand])
        assert len(results) == 0
        assert tel["malformed_records"] == 1

    def test_blank_lines_skipped(self):
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", mode="w", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write("\n")
            tmp.write(json.dumps(_make_candidate()) + "\n")
            tmp.write("\n\n")
            tmp_path = tmp.name

        try:
            filt = IntegrityFilter()
            results = list(filt.process_stream(tmp_path))
            assert len(results) == 1
        finally:
            os.unlink(tmp_path)

    def test_multiple_candidates_streamed(self):
        candidates = [
            _make_candidate(candidate_id=f"CAND_{i:04d}") for i in range(50)
        ]
        results, tel = _run_filter(candidates)
        assert len(results) == 50
        assert tel["total_candidates"] == 50
        assert tel["clean_passed"] == 50

    def test_streaming_yields_one_at_a_time(self):
        """Verify that process_stream is a true generator (not list)."""
        suffix = ".jsonl.gz"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
        candidates = [
            _make_candidate(candidate_id=f"CAND_{i:04d}") for i in range(5)
        ]
        _write_jsonl_gz(candidates, tmp_path)

        try:
            filt = IntegrityFilter()
            gen = filt.process_stream(tmp_path)
            # Consume one result — the rest should still be pending
            first = next(gen)
            assert first["candidate_id"] == "CAND_0000"
            # Consume the rest
            remaining = list(gen)
            assert len(remaining) == 4
        finally:
            os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Telemetry
# ═══════════════════════════════════════════════════════════════════════════


class TestTelemetry:
    def test_telemetry_has_all_keys(self):
        _, tel = _run_filter([_make_candidate()])
        expected_keys = {
            "run_timestamp",
            "duration_seconds",
            "total_candidates",
            "honeypots_detected",
            "clean_passed",
            "warnings_emitted",
            "honeypot_rate",
            "warning_rate",
            "date_parse_errors",
            "malformed_records",
            "check_stats",
        }
        assert expected_keys.issubset(set(tel.keys()))

    def test_check_stats_has_all_checks(self):
        _, tel = _run_filter([_make_candidate()])
        for name in [
            "experience_vs_education",
            "skill_duration",
            "future_dates",
            "career_consistency",
            "missing_fields",
        ]:
            assert name in tel["check_stats"]

    def test_check_stat_keys(self):
        _, tel = _run_filter([_make_candidate()])
        for stat in tel["check_stats"].values():
            assert "candidates_evaluated" in stat
            assert "candidates_skipped" in stat
            assert "flags_raised" in stat
            assert "warnings_raised" in stat

    def test_honeypot_rate_calculated(self):
        candidates = [
            _make_candidate(candidate_id="CLEAN_1"),
            _make_candidate(candidate_id="CLEAN_2"),
            _make_candidate(candidate_id="CLEAN_3"),
            {
                "candidate_id": "HONEYPOT_1",
                "profile": {"years_of_experience": 50},
                "education": [{"end_date": "2018-05-01"}],
                "skills": [{"name": "Python", "duration_months": 48}],
                "career_history": [],
            },
        ]
        _, tel = _run_filter(candidates)
        assert tel["honeypots_detected"] == 1
        assert tel["clean_passed"] == 3
        assert tel["honeypot_rate"] == pytest.approx(0.25, abs=0.001)

    def test_warning_rate_calculated(self):
        candidates = [
            _make_candidate(candidate_id="FULL_1"),
            {"candidate_id": "SPARSE_1"},  # 3 warnings (missing yoe, edu, skills)
        ]
        _, tel = _run_filter(candidates)
        assert tel["warnings_emitted"] == 3
        # 3 warnings / 2 candidates = 1.5
        assert tel["warning_rate"] == pytest.approx(1.5, abs=0.01)

    def test_write_telemetry_to_file(self):
        gz_path = None
        report_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".jsonl.gz", delete=False
            ) as tmp_gz:
                gz_path = tmp_gz.name
            _write_jsonl_gz([_make_candidate()], gz_path)

            with tempfile.NamedTemporaryFile(
                suffix=".json", delete=False
            ) as tmp_report:
                report_path = tmp_report.name

            filt = IntegrityFilter()
            list(filt.process_stream(gz_path))
            filt.write_telemetry(report_path)

            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["total_candidates"] == 1
            assert data["clean_passed"] == 1
        finally:
            if gz_path:
                os.unlink(gz_path)
            if report_path:
                os.unlink(report_path)

    def test_duration_is_positive(self):
        _, tel = _run_filter([_make_candidate()])
        assert tel["duration_seconds"] >= 0


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Date parsing edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestDateParsing:
    def test_malformed_date_increments_parse_errors(self):
        cand = _make_candidate()
        cand["education"] = [{"end_date": "not-a-date"}]
        _, tel = _run_filter([cand])
        assert tel["date_parse_errors"] >= 1

    def test_empty_string_date_increments_parse_errors(self):
        cand = _make_candidate()
        cand["education"] = [{"end_date": ""}]
        _, tel = _run_filter([cand])
        assert tel["date_parse_errors"] >= 1

    def test_numeric_date_handled(self):
        """Numeric value that cannot be a valid ISO date → parse error."""
        cand = _make_candidate()
        # 999 → str "999" is not a valid ISO date
        cand["education"] = [{"end_date": 999}]
        _, tel = _run_filter([cand])
        assert tel["date_parse_errors"] >= 1

    def test_none_date_returns_none_no_error(self):
        """Explicit None → _parse_date returns None, no error counter."""
        cand = _make_candidate()
        cand["career_history"] = [
            {
                "start_date": "2024-01-01",
                "end_date": None,
                "duration_months": 17,
            }
        ]
        _, tel = _run_filter([cand])
        # None dates should NOT increment parse errors
        assert tel["date_parse_errors"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Integration / end-to-end
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_schema_example_candidate(self):
        """The exact candidate from the provided schema passes cleanly."""
        cand = {
            "candidate_id": "CAND_0000001",
            "profile": {"years_of_experience": 6.9},
            "career_history": [
                {
                    "start_date": "2024-03-08",
                    "end_date": "2026-05-01",
                    "duration_months": 27,
                }
            ],
            "education": [{"end_date": "2018-05-01"}],
            "skills": [{"name": "Python", "duration_months": 48}],
        }
        results, tel = _run_filter([cand])
        assert len(results) == 1
        assert results[0]["is_honeypot"] is False
        assert results[0]["anomalies"] == []
        assert tel["honeypots_detected"] == 0
        assert tel["clean_passed"] == 1

    def test_mixed_batch_correct_counts(self):
        """Mix of clean and honeypot candidates → correct tallies."""
        clean = _make_candidate(candidate_id="CLEAN_1")
        honeypot = _make_candidate(candidate_id="HP_1")
        honeypot["profile"]["years_of_experience"] = 99
        sparse = {"candidate_id": "SPARSE_1"}  # 3 warnings, not honeypot

        results, tel = _run_filter([clean, honeypot, sparse])
        assert tel["total_candidates"] == 3
        assert tel["honeypots_detected"] == 1
        assert tel["clean_passed"] == 2
        assert tel["warnings_emitted"] == 3

        hp_result = next(r for r in results if r["candidate_id"] == "HP_1")
        assert hp_result["is_honeypot"] is True

        sparse_result = next(
            r for r in results if r["candidate_id"] == "SPARSE_1"
        )
        assert sparse_result["is_honeypot"] is False
