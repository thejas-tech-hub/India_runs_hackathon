"""
Stage 0B — HireabilityEvaluator Test Suite
============================================
Team: AlgoRhythms | Author: Thejas J

Covers:
    1. Schema sample candidate (golden path)
    2. Missing-data pessimistic candidate
    3. Perfect / best-case candidate
    4. Worst-case candidate
    5. Division / clamping edge cases
    6. Tier boundary tests
    7. Config validation
    8. Batch evaluation
    9. Telemetry verification
"""

import unittest
from datetime import date

from hireability_evaluator import (
    HireabilityConfig,
    HireabilityEvaluator,
    HireabilityResult,
)


# ═══════════════════════════════════════════════════════════════════
# Test fixtures
# ═══════════════════════════════════════════════════════════════════


def _schema_sample_candidate() -> dict:
    """The exact candidate from the provided schema (source of truth)."""
    return {
        "candidate_id": "CAND_0000001",
        "verified_email": True,
        "redrob_signals": {
            "profile_completeness_score": 85.5,
            "last_active_date": "2025-10-22",
            "recruiter_response_rate": 0.42,
            "interview_completion_rate": 0.80,
            "skill_assessment_scores": {"Python": 88.0},
            "endorsements_received": 22,
            "notice_period_days": 90,
            "github_activity_score": 75.0,
        },
    }


def _missing_data_candidate() -> dict:
    """Only candidate_id present — everything else missing."""
    return {"candidate_id": "CAND_MISSING"}


def _perfect_candidate() -> dict:
    """Best possible scores across all dimensions."""
    return {
        "candidate_id": "CAND_PERFECT",
        "verified_email": True,
        "redrob_signals": {
            "profile_completeness_score": 100.0,
            "last_active_date": "2026-06-08",
            "recruiter_response_rate": 0.95,
            "interview_completion_rate": 1.00,
            "skill_assessment_scores": {"Python": 95.0, "SQL": 90.0},
            "endorsements_received": 50,
            "notice_period_days": 15,
            "github_activity_score": 90.0,
        },
    }


def _worst_case_candidate() -> dict:
    """Worst possible scores — everything unfavorable."""
    return {
        "candidate_id": "CAND_WORST",
        "verified_email": False,
        "redrob_signals": {
            "profile_completeness_score": 0.0,
            "last_active_date": "2024-01-01",
            "recruiter_response_rate": 0.0,
            "interview_completion_rate": 0.0,
            "skill_assessment_scores": {},
            "endorsements_received": 0,
            "notice_period_days": 180,
            "github_activity_score": -1.0,
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════


class TestSchemaSampleCandidate(unittest.TestCase):
    """Test against the exact schema sample from the design document."""

    def setUp(self) -> None:
        self.evaluator = HireabilityEvaluator()
        self.result = self.evaluator.evaluate(_schema_sample_candidate())

    def test_candidate_id(self) -> None:
        self.assertEqual(self.result.candidate_id, "CAND_0000001")

    def test_availability_score(self) -> None:
        # NPM: 90 days → 61-90 bucket → 0.65
        # LAM: 2025-10-22 → 230 days inactive → ≤365 bucket → 0.40
        # Availability = (0.5 * 0.65 + 0.5 * 0.40) / 1.0 = 0.525
        self.assertAlmostEqual(self.result.availability_score, 0.525, places=3)

    def test_evidence_coverage_score(self) -> None:
        # C1: verified_email = True → +0.20
        # C2: github = 75.0 ≠ -1 → +0.25
        # C3: skills non-empty → +0.30
        # C4: endorsements 22 > 5 → +0.15
        # C5: profile 85.5/100 * 0.10 = 0.0855
        # Total = 0.9855
        self.assertAlmostEqual(
            self.result.evidence_coverage_score, 0.9855, places=3
        )

    def test_risk_score(self) -> None:
        # Ghosting: 1 - 0.42 = 0.58
        # Flake:    1 - 0.80 = 0.20
        # Timeline: 1 - 0.65 = 0.35
        # Risk = 0.45*0.58 + 0.35*0.20 + 0.20*0.35 = 0.401
        self.assertAlmostEqual(self.result.risk_score, 0.401, places=3)

    def test_components_notice_period_multiplier(self) -> None:
        self.assertAlmostEqual(
            self.result.components["notice_period_multiplier"], 0.65, places=4
        )

    def test_components_last_active_multiplier(self) -> None:
        self.assertAlmostEqual(
            self.result.components["last_active_multiplier"], 0.40, places=4
        )

    def test_components_ghosting_risk(self) -> None:
        self.assertAlmostEqual(
            self.result.components["ghosting_risk"], 0.58, places=4
        )

    def test_components_flake_risk(self) -> None:
        self.assertAlmostEqual(
            self.result.components["flake_risk"], 0.20, places=4
        )

    def test_components_timeline_risk(self) -> None:
        self.assertAlmostEqual(
            self.result.components["timeline_risk"], 0.35, places=4
        )

    def test_evidence_breakdown_verified_email(self) -> None:
        bd = self.result.components["evidence_breakdown"]
        self.assertEqual(bd["verified_email"], 0.20)

    def test_evidence_breakdown_github(self) -> None:
        bd = self.result.components["evidence_breakdown"]
        self.assertEqual(bd["github_activity"], 0.25)

    def test_evidence_breakdown_skills(self) -> None:
        bd = self.result.components["evidence_breakdown"]
        self.assertEqual(bd["skill_assessments"], 0.30)

    def test_evidence_breakdown_endorsements(self) -> None:
        bd = self.result.components["evidence_breakdown"]
        self.assertEqual(bd["endorsements"], 0.15)

    def test_evidence_breakdown_profile_completeness(self) -> None:
        bd = self.result.components["evidence_breakdown"]
        self.assertAlmostEqual(bd["profile_completeness"], 0.0855, places=4)

    def test_no_defaults_applied(self) -> None:
        self.assertEqual(self.result.telemetry["defaults_applied"], [])

    def test_no_warnings(self) -> None:
        self.assertEqual(self.result.telemetry["warnings"], [])

    def test_no_clamped_fields(self) -> None:
        self.assertEqual(self.result.telemetry["clamped_fields"], [])

    def test_input_hash_present(self) -> None:
        self.assertIsInstance(self.result.telemetry["input_hash"], str)
        self.assertEqual(len(self.result.telemetry["input_hash"]), 64)

    def test_to_dict_roundtrip(self) -> None:
        d = self.result.to_dict()
        self.assertEqual(d["candidate_id"], "CAND_0000001")
        self.assertIn("availability_score", d)
        self.assertIn("evidence_coverage_score", d)
        self.assertIn("risk_score", d)
        self.assertIn("components", d)
        self.assertIn("telemetry", d)


class TestMissingDataCandidate(unittest.TestCase):
    """All fields missing except candidate_id — pessimistic defaults."""

    def setUp(self) -> None:
        self.evaluator = HireabilityEvaluator()
        self.result = self.evaluator.evaluate(_missing_data_candidate())

    def test_candidate_id(self) -> None:
        self.assertEqual(self.result.candidate_id, "CAND_MISSING")

    def test_availability_score(self) -> None:
        # NPM: default 90 → 0.65
        # LAM: default 365 days → 0.40
        # Availability = 0.525
        self.assertAlmostEqual(self.result.availability_score, 0.525, places=3)

    def test_evidence_coverage_score_is_zero(self) -> None:
        # All evidence fields default to non-qualifying values
        self.assertAlmostEqual(
            self.result.evidence_coverage_score, 0.0, places=4
        )

    def test_risk_score_pessimistic(self) -> None:
        # Ghosting: 1 - 0.0 = 1.0
        # Flake:    1 - 0.0 = 1.0
        # Timeline: 1 - 0.65 = 0.35
        # Risk = 0.45*1.0 + 0.35*1.0 + 0.20*0.35 = 0.87
        self.assertAlmostEqual(self.result.risk_score, 0.87, places=3)

    def test_defaults_applied_recorded(self) -> None:
        defaults = self.result.telemetry["defaults_applied"]
        # redrob_signals itself is missing, plus all sub-fields and
        # verified_email
        self.assertIn("redrob_signals", defaults)
        self.assertIn("notice_period_days", defaults)
        self.assertIn("last_active_date", defaults)
        self.assertIn("recruiter_response_rate", defaults)
        self.assertIn("interview_completion_rate", defaults)
        self.assertIn("skill_assessment_scores", defaults)
        self.assertIn("endorsements_received", defaults)
        self.assertIn("profile_completeness_score", defaults)
        self.assertIn("github_activity_score", defaults)
        self.assertIn("verified_email", defaults)

    def test_all_scores_in_range(self) -> None:
        for score in [
            self.result.availability_score,
            self.result.evidence_coverage_score,
            self.result.risk_score,
        ]:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestPerfectCandidate(unittest.TestCase):
    """Best-case candidate across all dimensions."""

    def setUp(self) -> None:
        self.evaluator = HireabilityEvaluator()
        self.result = self.evaluator.evaluate(_perfect_candidate())

    def test_availability_score(self) -> None:
        # NPM: 15 days → ≤30 → 1.00
        # LAM: 2026-06-08 → 1 day → ≤30 → 1.00
        # Availability = 1.00
        self.assertAlmostEqual(self.result.availability_score, 1.0, places=3)

    def test_evidence_coverage_score_max(self) -> None:
        # All 5 components contribute maximum
        # 0.20 + 0.25 + 0.30 + 0.15 + (100/100)*0.10 = 1.00
        self.assertAlmostEqual(
            self.result.evidence_coverage_score, 1.0, places=3
        )

    def test_risk_score_near_zero(self) -> None:
        # Ghosting: 1 - 0.95 = 0.05
        # Flake:    1 - 1.00 = 0.00
        # Timeline: 1 - 1.00 = 0.00
        # Risk = 0.45*0.05 + 0 + 0 = 0.0225
        self.assertAlmostEqual(self.result.risk_score, 0.0225, places=3)

    def test_no_defaults_applied(self) -> None:
        self.assertEqual(self.result.telemetry["defaults_applied"], [])


class TestWorstCaseCandidate(unittest.TestCase):
    """Worst-case candidate — everything unfavorable."""

    def setUp(self) -> None:
        self.evaluator = HireabilityEvaluator()
        self.result = self.evaluator.evaluate(_worst_case_candidate())

    def test_availability_score(self) -> None:
        # NPM: 180 → >120 → floor 0.20
        # LAM: 2024-01-01 → ~525 days → >365 → floor 0.15
        # Availability = (0.5*0.20 + 0.5*0.15) / 1.0 = 0.175
        self.assertAlmostEqual(self.result.availability_score, 0.175, places=3)

    def test_evidence_coverage_score_zero(self) -> None:
        # verified_email=False, github=-1, skills={}, endorsements=0,
        # profile=0.0
        self.assertAlmostEqual(
            self.result.evidence_coverage_score, 0.0, places=4
        )

    def test_risk_score_extreme(self) -> None:
        # Ghosting: 1 - 0.0 = 1.0
        # Flake:    1 - 0.0 = 1.0
        # Timeline: 1 - 0.20 = 0.80
        # Risk = 0.45*1.0 + 0.35*1.0 + 0.20*0.80 = 0.96
        self.assertAlmostEqual(self.result.risk_score, 0.96, places=3)


class TestEdgeCases(unittest.TestCase):
    """Edge cases: missing ID, clamping, date parsing, boundaries."""

    def setUp(self) -> None:
        self.evaluator = HireabilityEvaluator()

    # ── E1: Missing candidate_id ─────────────────────────────────

    def test_missing_candidate_id_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.evaluator.evaluate({"verified_email": True})
        self.assertIn("candidate_id", str(ctx.exception))

    def test_none_candidate_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(
                {"candidate_id": None, "verified_email": True}
            )

    # ── E2: Negative notice period → clamp to 0 ─────────────────

    def test_negative_notice_period(self) -> None:
        candidate = {
            "candidate_id": "CAND_NEG_NP",
            "redrob_signals": {"notice_period_days": -5},
        }
        result = self.evaluator.evaluate(candidate)
        # -5 clamped to 0, 0 ≤ 30 → NPM = 1.00
        self.assertAlmostEqual(
            result.components["notice_period_multiplier"], 1.0, places=4
        )

    def test_zero_notice_period(self) -> None:
        candidate = {
            "candidate_id": "CAND_ZERO_NP",
            "redrob_signals": {"notice_period_days": 0},
        }
        result = self.evaluator.evaluate(candidate)
        self.assertAlmostEqual(
            result.components["notice_period_multiplier"], 1.0, places=4
        )

    # ── E3: Future last_active_date ──────────────────────────────

    def test_future_last_active_date(self) -> None:
        candidate = {
            "candidate_id": "CAND_FUTURE",
            "redrob_signals": {"last_active_date": "2027-01-01"},
        }
        result = self.evaluator.evaluate(candidate)
        # Future → 0 inactive days → ≤30 → LAM = 1.00
        self.assertAlmostEqual(
            result.components["last_active_multiplier"], 1.0, places=4
        )
        self.assertTrue(
            any("future" in w for w in result.telemetry["warnings"])
        )

    # ── E4: Unparseable last_active_date ─────────────────────────

    def test_unparseable_date(self) -> None:
        candidate = {
            "candidate_id": "CAND_BADDATE",
            "redrob_signals": {"last_active_date": "not-a-date"},
        }
        result = self.evaluator.evaluate(candidate)
        # Unparseable → default 365 days → ≤365 → LAM = 0.40
        self.assertAlmostEqual(
            result.components["last_active_multiplier"], 0.40, places=4
        )
        self.assertTrue(
            any("could not be parsed" in w for w in result.telemetry["warnings"])
        )

    def test_invalid_date_format(self) -> None:
        candidate = {
            "candidate_id": "CAND_BADFORMAT",
            "redrob_signals": {"last_active_date": "22/10/2025"},
        }
        result = self.evaluator.evaluate(candidate)
        self.assertAlmostEqual(
            result.components["last_active_multiplier"], 0.40, places=4
        )

    # ── E5: profile_completeness_score > 100 ─────────────────────

    def test_profile_completeness_over_100(self) -> None:
        candidate = {
            "candidate_id": "CAND_OVER100",
            "redrob_signals": {"profile_completeness_score": 150.0},
        }
        result = self.evaluator.evaluate(candidate)
        # Clamped: 150/100 → clamped to 1.0 → credit = 0.10
        bd = result.components["evidence_breakdown"]
        self.assertAlmostEqual(bd["profile_completeness"], 0.10, places=4)
        self.assertTrue(
            any(
                "profile_completeness_score clamped" in c
                for c in result.telemetry["clamped_fields"]
            )
        )

    def test_profile_completeness_negative(self) -> None:
        candidate = {
            "candidate_id": "CAND_NEG_PCS",
            "redrob_signals": {"profile_completeness_score": -20.0},
        }
        result = self.evaluator.evaluate(candidate)
        # Clamped: -20/100 → clamped to 0.0 → credit = 0.0
        bd = result.components["evidence_breakdown"]
        self.assertAlmostEqual(bd["profile_completeness"], 0.0, places=4)

    # ── E6: Negative endorsements → treated as 0 ────────────────

    def test_negative_endorsements(self) -> None:
        candidate = {
            "candidate_id": "CAND_NEG_END",
            "redrob_signals": {"endorsements_received": -3},
        }
        result = self.evaluator.evaluate(candidate)
        bd = result.components["evidence_breakdown"]
        self.assertEqual(bd["endorsements"], 0.0)

    # ── E11: github_activity_score == 0.0 earns credit ───────────

    def test_github_score_zero_earns_credit(self) -> None:
        candidate = {
            "candidate_id": "CAND_GH_ZERO",
            "redrob_signals": {"github_activity_score": 0.0},
        }
        result = self.evaluator.evaluate(candidate)
        bd = result.components["evidence_breakdown"]
        # 0.0 ≠ -1 → earns credit
        self.assertEqual(bd["github_activity"], 0.25)

    def test_github_sentinel_no_credit(self) -> None:
        candidate = {
            "candidate_id": "CAND_GH_SENT",
            "redrob_signals": {"github_activity_score": -1.0},
        }
        result = self.evaluator.evaluate(candidate)
        bd = result.components["evidence_breakdown"]
        self.assertEqual(bd["github_activity"], 0.0)

    # ── E12: verified_email is None → treated as False ───────────

    def test_verified_email_none(self) -> None:
        candidate = {
            "candidate_id": "CAND_VE_NONE",
            "verified_email": None,
        }
        result = self.evaluator.evaluate(candidate)
        bd = result.components["evidence_breakdown"]
        self.assertEqual(bd["verified_email"], 0.0)
        self.assertIn(
            "verified_email", result.telemetry["defaults_applied"]
        )

    # ── Rates above 1.0 → clamping ──────────────────────────────

    def test_response_rate_above_1_clamped(self) -> None:
        candidate = {
            "candidate_id": "CAND_RATE_HIGH",
            "redrob_signals": {
                "recruiter_response_rate": 1.5,
                "interview_completion_rate": 2.0,
            },
        }
        result = self.evaluator.evaluate(candidate)
        # Clamped to 1.0 → ghosting_risk = 0.0, flake_risk = 0.0
        self.assertAlmostEqual(
            result.components["ghosting_risk"], 0.0, places=4
        )
        self.assertAlmostEqual(
            result.components["flake_risk"], 0.0, places=4
        )
        clamped = result.telemetry["clamped_fields"]
        self.assertTrue(
            any("recruiter_response_rate" in c for c in clamped)
        )
        self.assertTrue(
            any("interview_completion_rate" in c for c in clamped)
        )

    # ── Redrob signals is not a dict ─────────────────────────────

    def test_redrob_signals_not_dict(self) -> None:
        candidate = {
            "candidate_id": "CAND_BAD_REDROB",
            "redrob_signals": "not_a_dict",
        }
        result = self.evaluator.evaluate(candidate)
        self.assertTrue(
            any("not a dict" in w for w in result.telemetry["warnings"])
        )
        # Should still produce valid scores with all defaults
        self.assertGreaterEqual(result.availability_score, 0.0)
        self.assertLessEqual(result.availability_score, 1.0)

    # ── skill_assessment_scores is not a dict ────────────────────

    def test_skill_scores_not_dict(self) -> None:
        candidate = {
            "candidate_id": "CAND_BAD_SKILLS",
            "redrob_signals": {"skill_assessment_scores": [1, 2, 3]},
        }
        result = self.evaluator.evaluate(candidate)
        bd = result.components["evidence_breakdown"]
        self.assertEqual(bd["skill_assessments"], 0.0)


class TestTierBoundaries(unittest.TestCase):
    """Exact boundary testing for notice period and last-active tiers."""

    def setUp(self) -> None:
        self.evaluator = HireabilityEvaluator()

    # ── Notice Period Tier Boundaries ────────────────────────────

    def _npm(self, days: int) -> float:
        """Shortcut to compute NPM for a given notice period."""
        candidate = {
            "candidate_id": "CAND_NPM_TEST",
            "redrob_signals": {"notice_period_days": days},
        }
        result = self.evaluator.evaluate(candidate)
        return result.components["notice_period_multiplier"]

    def test_np_30_boundary(self) -> None:
        self.assertAlmostEqual(self._npm(30), 1.00, places=4)

    def test_np_31_boundary(self) -> None:
        self.assertAlmostEqual(self._npm(31), 0.85, places=4)

    def test_np_60_boundary(self) -> None:
        self.assertAlmostEqual(self._npm(60), 0.85, places=4)

    def test_np_61_boundary(self) -> None:
        self.assertAlmostEqual(self._npm(61), 0.65, places=4)

    def test_np_90_boundary(self) -> None:
        self.assertAlmostEqual(self._npm(90), 0.65, places=4)

    def test_np_91_boundary(self) -> None:
        self.assertAlmostEqual(self._npm(91), 0.40, places=4)

    def test_np_120_boundary(self) -> None:
        self.assertAlmostEqual(self._npm(120), 0.40, places=4)

    def test_np_121_boundary(self) -> None:
        self.assertAlmostEqual(self._npm(121), 0.20, places=4)

    # ── Endorsement Boundary (> 5 required, not >= 5) ────────────

    def test_endorsements_exactly_5_no_credit(self) -> None:
        candidate = {
            "candidate_id": "CAND_END_5",
            "redrob_signals": {"endorsements_received": 5},
        }
        result = self.evaluator.evaluate(candidate)
        bd = result.components["evidence_breakdown"]
        self.assertEqual(bd["endorsements"], 0.0)

    def test_endorsements_6_gets_credit(self) -> None:
        candidate = {
            "candidate_id": "CAND_END_6",
            "redrob_signals": {"endorsements_received": 6},
        }
        result = self.evaluator.evaluate(candidate)
        bd = result.components["evidence_breakdown"]
        self.assertEqual(bd["endorsements"], 0.15)


class TestConfigValidation(unittest.TestCase):
    """Validate HireabilityConfig.validate() catches bad configs."""

    def test_default_config_is_valid(self) -> None:
        config = HireabilityConfig()
        errors = config.validate()
        self.assertEqual(errors, [])

    def test_negative_weight_rejected(self) -> None:
        config = HireabilityConfig(risk_weight_ghosting=-0.1)
        errors = config.validate()
        self.assertTrue(any("risk_weight_ghosting" in e for e in errors))

    def test_evidence_credits_exceeding_1_rejected(self) -> None:
        config = HireabilityConfig(evidence_verified_email_credit=0.80)
        # 0.80 + 0.25 + 0.30 + 0.15 + 0.10 = 1.60 > 1.0
        errors = config.validate()
        self.assertTrue(any("Evidence credits" in e for e in errors))

    def test_risk_weights_exceeding_1_rejected(self) -> None:
        config = HireabilityConfig(
            risk_weight_ghosting=0.50,
            risk_weight_flake=0.50,
            risk_weight_timeline=0.50,
        )
        errors = config.validate()
        self.assertTrue(any("Risk weights" in e for e in errors))

    def test_unsorted_tiers_rejected(self) -> None:
        config = HireabilityConfig(
            notice_period_tiers=[(90, 0.65), (30, 1.00), (60, 0.85)]
        )
        errors = config.validate()
        self.assertTrue(any("notice_period_tiers" in e for e in errors))

    def test_invalid_config_raises_on_evaluator_init(self) -> None:
        config = HireabilityConfig(risk_weight_ghosting=-1.0)
        with self.assertRaises(ValueError) as ctx:
            HireabilityEvaluator(config)
        self.assertIn("Invalid HireabilityConfig", str(ctx.exception))


class TestBatchEvaluation(unittest.TestCase):
    """Test evaluate_batch processes multiple candidates."""

    def test_batch_returns_correct_count(self) -> None:
        evaluator = HireabilityEvaluator()
        candidates = [
            _schema_sample_candidate(),
            _missing_data_candidate(),
            _perfect_candidate(),
        ]
        results = evaluator.evaluate_batch(candidates)
        self.assertEqual(len(results), 3)

    def test_batch_preserves_order(self) -> None:
        evaluator = HireabilityEvaluator()
        candidates = [
            _perfect_candidate(),
            _worst_case_candidate(),
        ]
        results = evaluator.evaluate_batch(candidates)
        self.assertEqual(results[0].candidate_id, "CAND_PERFECT")
        self.assertEqual(results[1].candidate_id, "CAND_WORST")

    def test_batch_results_match_individual(self) -> None:
        evaluator = HireabilityEvaluator()
        candidate = _schema_sample_candidate()

        individual = evaluator.evaluate(candidate)
        batch = evaluator.evaluate_batch([candidate])

        self.assertAlmostEqual(
            batch[0].availability_score,
            individual.availability_score,
            places=4,
        )
        self.assertAlmostEqual(
            batch[0].evidence_coverage_score,
            individual.evidence_coverage_score,
            places=4,
        )
        self.assertAlmostEqual(
            batch[0].risk_score,
            individual.risk_score,
            places=4,
        )

    def test_empty_batch(self) -> None:
        evaluator = HireabilityEvaluator()
        results = evaluator.evaluate_batch([])
        self.assertEqual(results, [])


class TestCustomConfig(unittest.TestCase):
    """Verify that custom config values are respected."""

    def test_custom_reference_date(self) -> None:
        config = HireabilityConfig(reference_date=date(2026, 6, 9))
        evaluator = HireabilityEvaluator(config)
        candidate = {
            "candidate_id": "CAND_CUSTOM",
            "redrob_signals": {"last_active_date": "2026-06-09"},
        }
        result = evaluator.evaluate(candidate)
        # Same day → 0 inactive days → ≤30 → LAM = 1.00
        self.assertAlmostEqual(
            result.components["last_active_multiplier"], 1.0, places=4
        )

    def test_custom_risk_weights(self) -> None:
        config = HireabilityConfig(
            risk_weight_ghosting=0.33,
            risk_weight_flake=0.34,
            risk_weight_timeline=0.33,
        )
        evaluator = HireabilityEvaluator(config)
        result = evaluator.evaluate(_schema_sample_candidate())
        # With equal-ish weights, risk differs from default
        self.assertIsNotNone(result.risk_score)
        self.assertGreaterEqual(result.risk_score, 0.0)
        self.assertLessEqual(result.risk_score, 1.0)

    def test_config_snapshot_in_telemetry(self) -> None:
        evaluator = HireabilityEvaluator()
        result = evaluator.evaluate(_schema_sample_candidate())
        snapshot = result.telemetry["config_snapshot"]
        self.assertEqual(snapshot["reference_date"], "2026-06-09")
        self.assertEqual(snapshot["risk_weights"]["ghosting"], 0.45)
        self.assertEqual(snapshot["risk_weights"]["flake"], 0.35)
        self.assertEqual(snapshot["risk_weights"]["timeline"], 0.20)
        self.assertEqual(snapshot["availability_weights"]["notice"], 0.50)
        self.assertEqual(snapshot["availability_weights"]["active"], 0.50)


class TestScoreBounds(unittest.TestCase):
    """Ensure all scores are always in [0.0, 1.0] regardless of input."""

    def setUp(self) -> None:
        self.evaluator = HireabilityEvaluator()

    def _assert_all_in_range(self, result: HireabilityResult) -> None:
        for name, score in [
            ("availability_score", result.availability_score),
            ("evidence_coverage_score", result.evidence_coverage_score),
            ("risk_score", result.risk_score),
        ]:
            self.assertGreaterEqual(
                score, 0.0, f"{name} below 0.0: {score}"
            )
            self.assertLessEqual(
                score, 1.0, f"{name} above 1.0: {score}"
            )

    def test_schema_sample_bounds(self) -> None:
        self._assert_all_in_range(
            self.evaluator.evaluate(_schema_sample_candidate())
        )

    def test_missing_data_bounds(self) -> None:
        self._assert_all_in_range(
            self.evaluator.evaluate(_missing_data_candidate())
        )

    def test_perfect_bounds(self) -> None:
        self._assert_all_in_range(
            self.evaluator.evaluate(_perfect_candidate())
        )

    def test_worst_case_bounds(self) -> None:
        self._assert_all_in_range(
            self.evaluator.evaluate(_worst_case_candidate())
        )

    def test_extreme_values_bounds(self) -> None:
        """Extreme out-of-range inputs still produce bounded scores."""
        candidate = {
            "candidate_id": "CAND_EXTREME",
            "verified_email": True,
            "redrob_signals": {
                "profile_completeness_score": 99999.0,
                "last_active_date": "1900-01-01",
                "recruiter_response_rate": -5.0,
                "interview_completion_rate": 100.0,
                "skill_assessment_scores": {"X": 0},
                "endorsements_received": 999999,
                "notice_period_days": 99999,
                "github_activity_score": 0.0,
            },
        }
        self._assert_all_in_range(self.evaluator.evaluate(candidate))


class TestHelperMethods(unittest.TestCase):
    """Direct tests of static helper methods."""

    def test_clamp_within_range(self) -> None:
        self.assertEqual(HireabilityEvaluator._clamp(0.5, 0.0, 1.0), 0.5)

    def test_clamp_below(self) -> None:
        self.assertEqual(HireabilityEvaluator._clamp(-1.0, 0.0, 1.0), 0.0)

    def test_clamp_above(self) -> None:
        self.assertEqual(HireabilityEvaluator._clamp(2.0, 0.0, 1.0), 1.0)

    def test_clamp_at_boundary_lo(self) -> None:
        self.assertEqual(HireabilityEvaluator._clamp(0.0, 0.0, 1.0), 0.0)

    def test_clamp_at_boundary_hi(self) -> None:
        self.assertEqual(HireabilityEvaluator._clamp(1.0, 0.0, 1.0), 1.0)

    def test_safe_get_existing_key(self) -> None:
        self.assertEqual(
            HireabilityEvaluator._safe_get({"a": 1}, "a", 99), 1
        )

    def test_safe_get_missing_key(self) -> None:
        self.assertEqual(
            HireabilityEvaluator._safe_get({"a": 1}, "b", 99), 99
        )

    def test_safe_get_none_value(self) -> None:
        self.assertEqual(
            HireabilityEvaluator._safe_get({"a": None}, "a", 99), 99
        )

    def test_safe_get_not_dict(self) -> None:
        self.assertEqual(
            HireabilityEvaluator._safe_get("not_a_dict", "a", 99), 99
        )

    def test_safe_get_false_value_not_defaulted(self) -> None:
        """False is a valid value and should NOT be replaced by default."""
        self.assertEqual(
            HireabilityEvaluator._safe_get({"a": False}, "a", True), False
        )

    def test_safe_get_zero_value_not_defaulted(self) -> None:
        """0 is a valid value and should NOT be replaced by default."""
        self.assertEqual(
            HireabilityEvaluator._safe_get({"a": 0}, "a", 99), 0
        )

    def test_tier_lookup_first_tier(self) -> None:
        tiers = [(30, 1.0), (60, 0.85), (90, 0.65)]
        self.assertEqual(
            HireabilityEvaluator._tier_lookup(10, tiers, 0.20), 1.0
        )

    def test_tier_lookup_middle_tier(self) -> None:
        tiers = [(30, 1.0), (60, 0.85), (90, 0.65)]
        self.assertEqual(
            HireabilityEvaluator._tier_lookup(45, tiers, 0.20), 0.85
        )

    def test_tier_lookup_floor(self) -> None:
        tiers = [(30, 1.0), (60, 0.85), (90, 0.65)]
        self.assertEqual(
            HireabilityEvaluator._tier_lookup(100, tiers, 0.20), 0.20
        )

    def test_tier_lookup_exact_boundary(self) -> None:
        tiers = [(30, 1.0), (60, 0.85), (90, 0.65)]
        self.assertEqual(
            HireabilityEvaluator._tier_lookup(30, tiers, 0.20), 1.0
        )


if __name__ == "__main__":
    unittest.main()
