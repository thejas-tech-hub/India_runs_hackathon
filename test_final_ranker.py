"""
Tests for Stage 2 — FinalRanker
=================================
Team:    AlgoRhythms
Student: THEJAS J

Coverage:
    1. Normal ranking with multiple candidates
    2. Single-candidate edge case (norm → 1.0)
    3. Zero-variance cross-encoder scores (all identical)
    4. Missing HireabilityResult entries (pessimistic defaults)
    5. Custom weight configurations
    6. Empty/invalid JD text rejection
    7. Empty candidates list rejection
    8. Composite score clamping (negative → 0.0)
    9. Stable sort tie-breaking verification
   10. Search document construction parity with Stage 1
   11. RankerConfig validation
   12. RankedCandidate immutability (frozen dataclass)
   13. to_dict() serialization
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hireability_evaluator import HireabilityResult
from final_ranker import (
    FinalRanker,
    RankedCandidate,
    RankerConfig,
    _NORM_EPSILON,
)


# ═══════════════════════════════════════════════════════════════════════════
# Test Fixtures & Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_candidate(
    candidate_id: str,
    headline: str = "Software Engineer",
    current_title: str = "Senior Developer",
    summary: str = "Experienced developer",
    skills: List[str] | None = None,
    career_history: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """Build a minimal candidate dict conforming to the schema."""
    if skills is None:
        skills = ["Python", "Java"]
    if career_history is None:
        career_history = [
            {"title": "Developer", "company": "TechCorp"},
        ]

    return {
        "candidate_id": candidate_id,
        "profile": {
            "headline": headline,
            "current_title": current_title,
            "summary": summary,
        },
        "skills": [{"name": s} for s in skills],
        "career_history": career_history,
    }


def _make_hireability_result(
    candidate_id: str,
    availability: float = 0.8,
    evidence: float = 0.7,
    risk: float = 0.2,
) -> HireabilityResult:
    """Build a HireabilityResult with controllable scores."""
    return HireabilityResult(
        candidate_id=candidate_id,
        availability_score=availability,
        evidence_coverage_score=evidence,
        risk_score=risk,
        components={},
        telemetry={},
    )


def _mock_cross_encoder_factory(scores: List[float]):
    """Return a mock CrossEncoder whose predict() returns the given scores.

    The mock is a class-level patch: calling CrossEncoder(name, device=...)
    returns an object whose .predict(pairs) returns np.array(scores).
    """
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array(scores, dtype=np.float64)
    return mock_model


# ═══════════════════════════════════════════════════════════════════════════
# RankerConfig Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRankerConfig:
    """Tests for RankerConfig validation."""

    def test_default_config_is_valid(self) -> None:
        """Default configuration passes validation."""
        config = RankerConfig()
        errors = config.validate()
        assert errors == []

    def test_negative_weight_rejected(self) -> None:
        """Negative weights produce validation errors."""
        config = RankerConfig(w_semantic=-0.1)
        errors = config.validate()
        assert len(errors) == 1
        assert "w_semantic" in errors[0]
        assert "non-negative" in errors[0]

    def test_additive_weights_exceeding_one_rejected(self) -> None:
        """Additive weights summing above 1.0 produce validation error."""
        config = RankerConfig(
            w_semantic=0.5, w_availability=0.3, w_evidence=0.3
        )
        errors = config.validate()
        assert len(errors) == 1
        assert "Additive weights" in errors[0]

    def test_risk_weight_exceeding_one_rejected(self) -> None:
        """Risk weight above 1.0 produces validation error."""
        config = RankerConfig(w_risk=1.5)
        errors = config.validate()
        assert len(errors) == 1
        assert "w_risk" in errors[0]

    def test_all_zero_weights_valid(self) -> None:
        """All-zero weights pass validation (degenerate but valid)."""
        config = RankerConfig(
            w_semantic=0.0,
            w_availability=0.0,
            w_evidence=0.0,
            w_risk=0.0,
        )
        errors = config.validate()
        assert errors == []

    def test_multiple_errors_reported(self) -> None:
        """Multiple validation failures are all reported."""
        config = RankerConfig(
            w_semantic=-0.1,
            w_risk=2.0,
        )
        errors = config.validate()
        assert len(errors) == 2


# ═══════════════════════════════════════════════════════════════════════════
# FinalRanker Initialization Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFinalRankerInit:
    """Tests for FinalRanker construction and config validation."""

    def test_default_config_accepted(self) -> None:
        """FinalRanker can be constructed with default config."""
        ranker = FinalRanker()
        assert ranker._config.w_semantic == 0.40

    def test_custom_config_accepted(self) -> None:
        """FinalRanker accepts a valid custom config."""
        config = RankerConfig(w_semantic=0.50, w_availability=0.20)
        ranker = FinalRanker(config=config)
        assert ranker._config.w_semantic == 0.50

    def test_invalid_config_raises_valueerror(self) -> None:
        """FinalRanker rejects an invalid config with ValueError."""
        config = RankerConfig(w_semantic=-1.0)
        with pytest.raises(ValueError, match="Invalid RankerConfig"):
            FinalRanker(config=config)


# ═══════════════════════════════════════════════════════════════════════════
# Input Validation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestInputValidation:
    """Tests for rank() input validation."""

    @patch("final_ranker.CrossEncoder")
    def test_empty_jd_raises_valueerror(self, mock_ce_cls: MagicMock) -> None:
        """Empty JD text raises ValueError."""
        ranker = FinalRanker()
        candidates = [_make_candidate("C001")]
        hr = {"C001": _make_hireability_result("C001")}
        with pytest.raises(ValueError, match="jd_text must be a non-empty"):
            ranker.rank("", candidates, hr)

    @patch("final_ranker.CrossEncoder")
    def test_whitespace_jd_raises_valueerror(
        self, mock_ce_cls: MagicMock
    ) -> None:
        """Whitespace-only JD text raises ValueError."""
        ranker = FinalRanker()
        candidates = [_make_candidate("C001")]
        hr = {"C001": _make_hireability_result("C001")}
        with pytest.raises(ValueError, match="jd_text must be a non-empty"):
            ranker.rank("   \n\t  ", candidates, hr)

    @patch("final_ranker.CrossEncoder")
    def test_empty_candidates_raises_valueerror(
        self, mock_ce_cls: MagicMock
    ) -> None:
        """Empty candidates list raises ValueError."""
        ranker = FinalRanker()
        with pytest.raises(ValueError, match="candidates list must not be"):
            ranker.rank("Some JD text", [], {})


# ═══════════════════════════════════════════════════════════════════════════
# MinMax Normalization Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestNormalization:
    """Tests for FinalRanker._normalize_scores static method."""

    def test_normal_range_normalization(self) -> None:
        """Scores with spread are correctly normalized to [0, 1]."""
        raw = [2.0, 5.0, 8.0]
        result = FinalRanker._normalize_scores(raw)
        assert result == pytest.approx([0.0, 0.5, 1.0])

    def test_two_scores_normalization(self) -> None:
        """Two distinct scores normalize to 0.0 and 1.0."""
        raw = [-3.0, 7.0]
        result = FinalRanker._normalize_scores(raw)
        assert result == pytest.approx([0.0, 1.0])

    def test_negative_scores_normalization(self) -> None:
        """Negative raw scores are handled correctly."""
        raw = [-10.0, -5.0, 0.0]
        result = FinalRanker._normalize_scores(raw)
        assert result == pytest.approx([0.0, 0.5, 1.0])

    def test_zero_variance_all_identical(self) -> None:
        """All identical scores → all receive 1.0 (zero-variance fallback)."""
        raw = [3.5, 3.5, 3.5, 3.5]
        result = FinalRanker._normalize_scores(raw)
        assert result == [1.0, 1.0, 1.0, 1.0]

    def test_zero_variance_single_candidate(self) -> None:
        """Single candidate receives 1.0 (degenerate zero-variance)."""
        raw = [7.2]
        result = FinalRanker._normalize_scores(raw)
        assert result == [1.0]

    def test_zero_variance_near_identical(self) -> None:
        """Scores within epsilon of each other → all 1.0."""
        epsilon = _NORM_EPSILON / 10  # well within epsilon
        raw = [5.0, 5.0 + epsilon, 5.0 - epsilon]
        result = FinalRanker._normalize_scores(raw)
        assert result == [1.0, 1.0, 1.0]

    def test_empty_list(self) -> None:
        """Empty input returns empty output."""
        result = FinalRanker._normalize_scores([])
        assert result == []

    def test_large_spread(self) -> None:
        """Large spread normalizes correctly."""
        raw = [-12.0, 0.0, 11.0]
        result = FinalRanker._normalize_scores(raw)
        assert result[0] == pytest.approx(0.0)
        assert result[2] == pytest.approx(1.0)
        assert result[1] == pytest.approx(12.0 / 23.0)


# ═══════════════════════════════════════════════════════════════════════════
# Search Document Construction Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildSearchDocument:
    """Tests for _build_search_document parity with Stage 1."""

    def test_full_candidate(self) -> None:
        """All fields present → concatenated in correct order."""
        candidate = _make_candidate(
            "C001",
            headline="ML Engineer",
            current_title="Staff ML Eng",
            summary="Deep learning expert",
            skills=["PyTorch", "TensorFlow"],
            career_history=[
                {"title": "ML Lead", "company": "DeepAI"},
                {"title": "Data Scientist", "company": "BigData Inc"},
            ],
        )
        doc = FinalRanker._build_search_document(candidate)
        assert "ML Engineer" in doc
        assert "Staff ML Eng" in doc
        assert "Deep learning expert" in doc
        assert "PyTorch" in doc
        assert "TensorFlow" in doc
        assert "ML Lead DeepAI" in doc
        assert "Data Scientist BigData Inc" in doc

    def test_missing_profile(self) -> None:
        """Missing profile → uses empty strings for profile fields."""
        candidate = {"candidate_id": "C002", "skills": [{"name": "Go"}]}
        doc = FinalRanker._build_search_document(candidate)
        assert "Go" in doc
        # No profile fields, but no crash
        assert "None" not in doc

    def test_missing_skills(self) -> None:
        """Missing skills → skill portion is empty, no crash."""
        candidate = {
            "candidate_id": "C003",
            "profile": {"headline": "Backend Dev"},
        }
        doc = FinalRanker._build_search_document(candidate)
        assert "Backend Dev" in doc

    def test_missing_career_history(self) -> None:
        """Missing career_history → career portion is empty, no crash."""
        candidate = {
            "candidate_id": "C004",
            "profile": {"summary": "Full-stack developer"},
        }
        doc = FinalRanker._build_search_document(candidate)
        assert "Full-stack developer" in doc

    def test_completely_empty_candidate(self) -> None:
        """Candidate with no relevant fields → empty string."""
        candidate = {"candidate_id": "C005"}
        doc = FinalRanker._build_search_document(candidate)
        assert doc == ""

    def test_none_skill_names_skipped(self) -> None:
        """Skills with None names are silently skipped."""
        candidate = {
            "candidate_id": "C006",
            "skills": [{"name": None}, {"name": "Rust"}, {}],
        }
        doc = FinalRanker._build_search_document(candidate)
        assert "Rust" in doc
        assert "None" not in doc

    def test_career_entry_with_only_company(self) -> None:
        """Career entry missing title → only company name in output."""
        candidate = {
            "candidate_id": "C007",
            "career_history": [{"company": "SoloCorp"}],
        }
        doc = FinalRanker._build_search_document(candidate)
        assert "SoloCorp" in doc


# ═══════════════════════════════════════════════════════════════════════════
# End-to-End Ranking Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRankEndToEnd:
    """End-to-end tests for the rank() method with mocked CrossEncoder."""

    @patch("final_ranker.CrossEncoder")
    def test_normal_ranking_three_candidates(
        self, mock_ce_cls: MagicMock
    ) -> None:
        """Three candidates ranked correctly by composite score."""
        # Cross-encoder scores: C001=2.0, C002=8.0, C003=5.0
        mock_model = _mock_cross_encoder_factory([2.0, 8.0, 5.0])
        mock_ce_cls.return_value = mock_model

        candidates = [
            _make_candidate("C001"),
            _make_candidate("C002"),
            _make_candidate("C003"),
        ]
        hr = {
            "C001": _make_hireability_result("C001", 0.8, 0.7, 0.2),
            "C002": _make_hireability_result("C002", 0.6, 0.5, 0.3),
            "C003": _make_hireability_result("C003", 0.9, 0.8, 0.1),
        }

        ranker = FinalRanker()
        results = ranker.rank("Senior Python developer", candidates, hr)

        # Verify we get 3 results
        assert len(results) == 3

        # Verify descending order
        assert results[0].final_score >= results[1].final_score
        assert results[1].final_score >= results[2].final_score

        # All scores in [0, 1]
        for r in results:
            assert 0.0 <= r.final_score <= 1.0

        # Verify the CrossEncoder was called with correct pairs
        mock_model.predict.assert_called_once()

    @patch("final_ranker.CrossEncoder")
    def test_single_candidate(self, mock_ce_cls: MagicMock) -> None:
        """Single candidate receives norm_semantic_score = 1.0."""
        mock_model = _mock_cross_encoder_factory([5.0])
        mock_ce_cls.return_value = mock_model

        candidates = [_make_candidate("C001")]
        hr = {"C001": _make_hireability_result("C001", 0.8, 0.7, 0.2)}

        ranker = FinalRanker()
        results = ranker.rank("Data engineer role", candidates, hr)

        assert len(results) == 1
        assert results[0].candidate_id == "C001"
        assert results[0].norm_semantic_score == 1.0

        # Verify composite: 0.40*1.0 + 0.25*0.8 + 0.20*0.7 - 0.15*0.2
        expected = 0.40 * 1.0 + 0.25 * 0.8 + 0.20 * 0.7 - 0.15 * 0.2
        assert results[0].final_score == pytest.approx(
            round(expected, 4), abs=1e-4
        )

    @patch("final_ranker.CrossEncoder")
    def test_zero_variance_scores(self, mock_ce_cls: MagicMock) -> None:
        """All identical cross-encoder scores → all norm = 1.0,
        behavioral scores break ties."""
        mock_model = _mock_cross_encoder_factory([5.0, 5.0, 5.0])
        mock_ce_cls.return_value = mock_model

        candidates = [
            _make_candidate("C001"),
            _make_candidate("C002"),
            _make_candidate("C003"),
        ]
        hr = {
            "C001": _make_hireability_result("C001", 0.5, 0.5, 0.5),
            "C002": _make_hireability_result("C002", 0.9, 0.9, 0.1),
            "C003": _make_hireability_result("C003", 0.1, 0.1, 0.9),
        }

        ranker = FinalRanker()
        results = ranker.rank("Any JD text", candidates, hr)

        # All should have norm_semantic_score = 1.0
        for r in results:
            assert r.norm_semantic_score == 1.0

        # C002 has best behavioral scores → should rank first
        assert results[0].candidate_id == "C002"
        # C003 has worst behavioral scores → should rank last
        assert results[-1].candidate_id == "C003"


# ═══════════════════════════════════════════════════════════════════════════
# Missing HireabilityResult Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMissingHireabilityResult:
    """Tests for pessimistic default behavior when HireabilityResult is absent."""

    @patch("final_ranker.CrossEncoder")
    def test_missing_hr_receives_pessimistic_defaults(
        self, mock_ce_cls: MagicMock
    ) -> None:
        """Candidate without HireabilityResult gets avail=0, evid=0, risk=1."""
        mock_model = _mock_cross_encoder_factory([8.0, 2.0])
        mock_ce_cls.return_value = mock_model

        candidates = [
            _make_candidate("C001"),  # has HR
            _make_candidate("C002"),  # missing HR
        ]
        hr = {
            "C001": _make_hireability_result("C001", 0.8, 0.7, 0.2),
            # C002 intentionally missing
        }

        ranker = FinalRanker()
        results = ranker.rank("Python developer", candidates, hr)

        # Find C002 in results
        c002 = next(r for r in results if r.candidate_id == "C002")
        assert c002.availability_score == 0.0
        assert c002.evidence_coverage_score == 0.0
        assert c002.risk_score == 1.0

    @patch("final_ranker.CrossEncoder")
    def test_missing_hr_telemetry_recorded(
        self, mock_ce_cls: MagicMock
    ) -> None:
        """Missing HR records telemetry about defaults applied."""
        mock_model = _mock_cross_encoder_factory([5.0])
        mock_ce_cls.return_value = mock_model

        candidates = [_make_candidate("C001")]
        hr: Dict[str, HireabilityResult] = {}  # no HR at all

        ranker = FinalRanker()
        results = ranker.rank("ML engineer", candidates, hr)

        c001 = results[0]
        assert len(c001.telemetry["defaults_applied"]) == 1
        default_info = c001.telemetry["defaults_applied"][0]
        assert default_info["reason"] == "no_hireability_result"
        assert default_info["defaults_used"]["availability_score"] == 0.0
        assert default_info["defaults_used"]["risk_score"] == 1.0

    @patch("final_ranker.CrossEncoder")
    def test_missing_hr_telemetry_has_warning(
        self, mock_ce_cls: MagicMock
    ) -> None:
        """Missing HR records a human-readable warning in telemetry."""
        mock_model = _mock_cross_encoder_factory([5.0])
        mock_ce_cls.return_value = mock_model

        candidates = [_make_candidate("C001")]
        hr: Dict[str, HireabilityResult] = {}

        ranker = FinalRanker()
        results = ranker.rank("Any role", candidates, hr)

        c001 = results[0]
        assert len(c001.telemetry["warnings"]) >= 1
        assert "pessimistic defaults" in c001.telemetry["warnings"][0]

    @patch("final_ranker.CrossEncoder")
    def test_missing_hr_max_score_is_025(
        self, mock_ce_cls: MagicMock
    ) -> None:
        """Best possible score for missing HR candidate is 0.25."""
        # Single candidate → norm_semantic = 1.0
        mock_model = _mock_cross_encoder_factory([10.0])
        mock_ce_cls.return_value = mock_model

        candidates = [_make_candidate("C001")]
        hr: Dict[str, HireabilityResult] = {}

        ranker = FinalRanker()
        results = ranker.rank("Top engineer", candidates, hr)

        # 0.40*1.0 + 0.25*0.0 + 0.20*0.0 - 0.15*1.0 = 0.25
        assert results[0].final_score == pytest.approx(0.25, abs=1e-4)

    @patch("final_ranker.CrossEncoder")
    def test_missing_hr_candidate_ranks_below_complete_candidate(
        self, mock_ce_cls: MagicMock
    ) -> None:
        """Even with higher semantic score, missing HR ranks below complete."""
        # C001 (missing HR) gets higher semantic score
        # C002 (has HR) gets lower semantic score
        mock_model = _mock_cross_encoder_factory([10.0, 2.0])
        mock_ce_cls.return_value = mock_model

        candidates = [
            _make_candidate("C001"),  # missing HR, semantic=1.0
            _make_candidate("C002"),  # has HR,     semantic=0.0
        ]
        hr = {
            "C002": _make_hireability_result("C002", 0.8, 0.7, 0.2),
        }

        ranker = FinalRanker()
        results = ranker.rank("Some role", candidates, hr)

        # C002 should rank above C001 despite lower semantic score
        # C001: 0.40*1.0 + 0 + 0 - 0.15*1.0 = 0.25
        # C002: 0.40*0.0 + 0.25*0.8 + 0.20*0.7 - 0.15*0.2 = 0.31
        assert results[0].candidate_id == "C002"
        assert results[1].candidate_id == "C001"


# ═══════════════════════════════════════════════════════════════════════════
# Composite Score Clamping Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCompositeClamping:
    """Tests for clamping the final composite score to [0.0, 1.0]."""

    @patch("final_ranker.CrossEncoder")
    def test_negative_composite_clamped_to_zero(
        self, mock_ce_cls: MagicMock
    ) -> None:
        """Candidate with low semantic and high risk clamps to 0.0."""
        # norm_semantic = 0.0 (worst in batch)
        # With pessimistic defaults: 0.40*0.0 + 0 + 0 - 0.15*1.0 = -0.15
        mock_model = _mock_cross_encoder_factory([0.0, 10.0])
        mock_ce_cls.return_value = mock_model

        candidates = [
            _make_candidate("C001"),  # semantic=0.0
            _make_candidate("C002"),  # semantic=1.0
        ]
        hr: Dict[str, HireabilityResult] = {
            # C001 has no HR → pessimistic defaults
            "C002": _make_hireability_result("C002", 0.5, 0.5, 0.5),
        }

        ranker = FinalRanker()
        results = ranker.rank("Engineer", candidates, hr)

        c001 = next(r for r in results if r.candidate_id == "C001")
        assert c001.final_score == 0.0
        assert c001.components["was_clamped"] is True

    @patch("final_ranker.CrossEncoder")
    def test_score_within_range_not_clamped(
        self, mock_ce_cls: MagicMock
    ) -> None:
        """Score within [0, 1] is not marked as clamped."""
        mock_model = _mock_cross_encoder_factory([5.0])
        mock_ce_cls.return_value = mock_model

        candidates = [_make_candidate("C001")]
        hr = {"C001": _make_hireability_result("C001", 0.5, 0.5, 0.5)}

        ranker = FinalRanker()
        results = ranker.rank("Developer", candidates, hr)

        assert results[0].components["was_clamped"] is False


# ═══════════════════════════════════════════════════════════════════════════
# Stable Sort Tie-Breaking Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestStableSortTieBreaking:
    """Tests verifying that candidates with identical final scores
    retain their original input order (Stage 1 ranking preserved)."""

    @patch("final_ranker.CrossEncoder")
    def test_identical_scores_preserve_input_order(
        self, mock_ce_cls: MagicMock
    ) -> None:
        """Candidates with identical semantic and behavioral scores
        maintain input order."""
        # All candidates get identical cross-encoder scores
        mock_model = _mock_cross_encoder_factory([5.0, 5.0, 5.0, 5.0])
        mock_ce_cls.return_value = mock_model

        candidates = [
            _make_candidate("C_FIRST"),
            _make_candidate("C_SECOND"),
            _make_candidate("C_THIRD"),
            _make_candidate("C_FOURTH"),
        ]
        # All identical behavioral scores
        hr = {
            cid: _make_hireability_result(cid, 0.5, 0.5, 0.3)
            for cid in ["C_FIRST", "C_SECOND", "C_THIRD", "C_FOURTH"]
        }

        ranker = FinalRanker()
        results = ranker.rank("Any role", candidates, hr)

        # All have the same final score — input order preserved
        ids = [r.candidate_id for r in results]
        assert ids == ["C_FIRST", "C_SECOND", "C_THIRD", "C_FOURTH"]

    @patch("final_ranker.CrossEncoder")
    def test_partial_ties_stable(self, mock_ce_cls: MagicMock) -> None:
        """When some candidates tie but others don't, order is correct."""
        # C001 and C003 will tie after composite; C002 is clearly better
        mock_model = _mock_cross_encoder_factory([3.0, 10.0, 3.0])
        mock_ce_cls.return_value = mock_model

        candidates = [
            _make_candidate("C001"),
            _make_candidate("C002"),
            _make_candidate("C003"),
        ]
        # Give C001 and C003 identical behavioral scores
        hr = {
            "C001": _make_hireability_result("C001", 0.5, 0.5, 0.3),
            "C002": _make_hireability_result("C002", 0.5, 0.5, 0.3),
            "C003": _make_hireability_result("C003", 0.5, 0.5, 0.3),
        }

        ranker = FinalRanker()
        results = ranker.rank("Engineer", candidates, hr)

        # C002 should be first (highest semantic); C001 before C003 (tie, stable)
        assert results[0].candidate_id == "C002"
        assert results[1].candidate_id == "C001"
        assert results[2].candidate_id == "C003"


# ═══════════════════════════════════════════════════════════════════════════
# Custom Weights Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCustomWeights:
    """Tests for custom weight configurations."""

    @patch("final_ranker.CrossEncoder")
    def test_semantic_only_weights(self, mock_ce_cls: MagicMock) -> None:
        """With only semantic weight, ranking is purely by cross-encoder."""
        mock_model = _mock_cross_encoder_factory([1.0, 5.0, 3.0])
        mock_ce_cls.return_value = mock_model

        config = RankerConfig(
            w_semantic=1.0,
            w_availability=0.0,
            w_evidence=0.0,
            w_risk=0.0,
        )
        ranker = FinalRanker(config=config)

        candidates = [
            _make_candidate("C001"),
            _make_candidate("C002"),
            _make_candidate("C003"),
        ]
        hr = {
            "C001": _make_hireability_result("C001", 1.0, 1.0, 0.0),
            "C002": _make_hireability_result("C002", 0.0, 0.0, 1.0),
            "C003": _make_hireability_result("C003", 0.5, 0.5, 0.5),
        }

        results = ranker.rank("Pure semantic test", candidates, hr)

        # Purely by semantic: C002(1.0) > C003(0.5) > C001(0.0)
        assert results[0].candidate_id == "C002"
        assert results[1].candidate_id == "C003"
        assert results[2].candidate_id == "C001"

    @patch("final_ranker.CrossEncoder")
    def test_behavioral_dominated_weights(
        self, mock_ce_cls: MagicMock
    ) -> None:
        """With very low semantic weight, behavioral scores dominate."""
        mock_model = _mock_cross_encoder_factory([10.0, 1.0])
        mock_ce_cls.return_value = mock_model

        config = RankerConfig(
            w_semantic=0.01,
            w_availability=0.40,
            w_evidence=0.40,
            w_risk=0.10,
        )
        ranker = FinalRanker(config=config)

        candidates = [
            _make_candidate("C001"),  # high semantic, bad behavioral
            _make_candidate("C002"),  # low semantic, good behavioral
        ]
        hr = {
            "C001": _make_hireability_result("C001", 0.1, 0.1, 0.9),
            "C002": _make_hireability_result("C002", 0.9, 0.9, 0.1),
        }

        results = ranker.rank("Any role", candidates, hr)

        # Behavioral dominates: C002 should rank first
        assert results[0].candidate_id == "C002"


# ═══════════════════════════════════════════════════════════════════════════
# Composite Score Calculation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCompositeCalculation:
    """Tests for _compute_composite method directly."""

    def test_perfect_candidate(self) -> None:
        """Perfect scores across all dimensions."""
        ranker = FinalRanker()
        result = ranker._compute_composite(
            norm_semantic=1.0,
            availability=1.0,
            evidence=1.0,
            risk=0.0,
        )
        # 0.40*1.0 + 0.25*1.0 + 0.20*1.0 - 0.15*0.0 = 0.85
        assert result == pytest.approx(0.85, abs=1e-6)

    def test_worst_candidate(self) -> None:
        """Worst possible scores across all dimensions."""
        ranker = FinalRanker()
        result = ranker._compute_composite(
            norm_semantic=0.0,
            availability=0.0,
            evidence=0.0,
            risk=1.0,
        )
        # 0.40*0.0 + 0.25*0.0 + 0.20*0.0 - 0.15*1.0 = -0.15
        assert result == pytest.approx(-0.15, abs=1e-6)

    def test_mid_range_candidate(self) -> None:
        """Mid-range scores produce expected composite."""
        ranker = FinalRanker()
        result = ranker._compute_composite(
            norm_semantic=0.5,
            availability=0.5,
            evidence=0.5,
            risk=0.5,
        )
        # 0.40*0.5 + 0.25*0.5 + 0.20*0.5 - 0.15*0.5
        # = 0.20 + 0.125 + 0.10 - 0.075 = 0.35
        assert result == pytest.approx(0.35, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# RankedCandidate Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRankedCandidate:
    """Tests for the RankedCandidate dataclass."""

    def test_frozen_immutability(self) -> None:
        """RankedCandidate is frozen — attribute assignment raises error."""
        rc = RankedCandidate(
            candidate_id="C001",
            final_score=0.75,
            norm_semantic_score=0.8,
            availability_score=0.7,
            evidence_coverage_score=0.6,
            risk_score=0.2,
            components={},
            telemetry={},
        )
        with pytest.raises(AttributeError):
            rc.final_score = 0.99  # type: ignore[misc]

    def test_to_dict_serialization(self) -> None:
        """to_dict() produces correctly rounded dictionary."""
        rc = RankedCandidate(
            candidate_id="C001",
            final_score=0.75123456,
            norm_semantic_score=0.81234,
            availability_score=0.71234,
            evidence_coverage_score=0.61234,
            risk_score=0.21234,
            components={"key": "value"},
            telemetry={"info": "data"},
        )
        d = rc.to_dict()
        assert d["candidate_id"] == "C001"
        assert d["final_score"] == 0.7512
        assert d["norm_semantic_score"] == 0.8123
        assert d["availability_score"] == 0.7123
        assert d["evidence_coverage_score"] == 0.6123
        assert d["risk_score"] == 0.2123
        assert d["components"] == {"key": "value"}
        assert d["telemetry"] == {"info": "data"}

    def test_to_dict_contains_all_keys(self) -> None:
        """to_dict() contains all expected keys."""
        rc = RankedCandidate(
            candidate_id="C001",
            final_score=0.5,
            norm_semantic_score=0.5,
            availability_score=0.5,
            evidence_coverage_score=0.5,
            risk_score=0.5,
            components={},
            telemetry={},
        )
        d = rc.to_dict()
        expected_keys = {
            "candidate_id",
            "final_score",
            "norm_semantic_score",
            "availability_score",
            "evidence_coverage_score",
            "risk_score",
            "components",
            "telemetry",
        }
        assert set(d.keys()) == expected_keys


# ═══════════════════════════════════════════════════════════════════════════
# CrossEncoder Integration Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossEncoderIntegration:
    """Tests verifying CrossEncoder is initialized and called correctly."""

    @patch("final_ranker.CrossEncoder")
    def test_model_loaded_with_cpu_device(
        self, mock_ce_cls: MagicMock
    ) -> None:
        """CrossEncoder is initialized with device='cpu'."""
        mock_model = _mock_cross_encoder_factory([5.0])
        mock_ce_cls.return_value = mock_model

        ranker = FinalRanker()
        candidates = [_make_candidate("C001")]
        hr = {"C001": _make_hireability_result("C001")}
        ranker.rank("Test JD", candidates, hr)

        mock_ce_cls.assert_called_once_with(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"
        )

    @patch("final_ranker.CrossEncoder")
    def test_model_loaded_once_on_multiple_calls(
        self, mock_ce_cls: MagicMock
    ) -> None:
        """CrossEncoder model is lazily loaded and cached across calls."""
        mock_model = _mock_cross_encoder_factory([5.0])
        mock_ce_cls.return_value = mock_model

        ranker = FinalRanker()
        candidates = [_make_candidate("C001")]
        hr = {"C001": _make_hireability_result("C001")}

        ranker.rank("First call", candidates, hr)
        ranker.rank("Second call", candidates, hr)

        # Model constructed only once despite two rank() calls
        assert mock_ce_cls.call_count == 1

    @patch("final_ranker.CrossEncoder")
    def test_predict_receives_correct_pairs(
        self, mock_ce_cls: MagicMock
    ) -> None:
        """predict() is called with [jd_text, search_doc] pairs."""
        mock_model = _mock_cross_encoder_factory([3.0, 7.0])
        mock_ce_cls.return_value = mock_model

        candidates = [
            _make_candidate("C001", headline="Frontend Dev"),
            _make_candidate("C002", headline="Backend Dev"),
        ]
        hr = {
            "C001": _make_hireability_result("C001"),
            "C002": _make_hireability_result("C002"),
        }

        ranker = FinalRanker()
        jd = "React developer needed"
        ranker.rank(jd, candidates, hr)

        # Verify the pairs passed to predict
        call_args = mock_model.predict.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0][0] == jd  # first pair starts with JD
        assert call_args[1][0] == jd  # second pair starts with JD
        assert "Frontend Dev" in call_args[0][1]
        assert "Backend Dev" in call_args[1][1]

    @patch("final_ranker.CrossEncoder")
    def test_custom_model_name_used(
        self, mock_ce_cls: MagicMock
    ) -> None:
        """Custom model_name in config is passed to CrossEncoder."""
        mock_model = _mock_cross_encoder_factory([5.0])
        mock_ce_cls.return_value = mock_model

        config = RankerConfig(model_name="custom/test-model")
        ranker = FinalRanker(config=config)
        candidates = [_make_candidate("C001")]
        hr = {"C001": _make_hireability_result("C001")}
        ranker.rank("Test", candidates, hr)

        mock_ce_cls.assert_called_once_with(
            "custom/test-model", device="cpu"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Component Breakdown Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestComponentBreakdown:
    """Tests verifying the components dict in RankedCandidate."""

    @patch("final_ranker.CrossEncoder")
    def test_components_contain_all_expected_keys(
        self, mock_ce_cls: MagicMock
    ) -> None:
        """Components dict has all expected intermediate values."""
        mock_model = _mock_cross_encoder_factory([5.0])
        mock_ce_cls.return_value = mock_model

        candidates = [_make_candidate("C001")]
        hr = {"C001": _make_hireability_result("C001", 0.8, 0.7, 0.2)}

        ranker = FinalRanker()
        results = ranker.rank("Test JD", candidates, hr)

        expected_keys = {
            "raw_semantic_score",
            "norm_semantic_score",
            "weighted_semantic",
            "weighted_availability",
            "weighted_evidence",
            "weighted_risk_penalty",
            "raw_composite",
            "was_clamped",
        }
        assert set(results[0].components.keys()) == expected_keys

    @patch("final_ranker.CrossEncoder")
    def test_weighted_components_match_formula(
        self, mock_ce_cls: MagicMock
    ) -> None:
        """Weighted components in breakdown match the formula."""
        mock_model = _mock_cross_encoder_factory([7.0])
        mock_ce_cls.return_value = mock_model

        candidates = [_make_candidate("C001")]
        hr = {"C001": _make_hireability_result("C001", 0.8, 0.6, 0.3)}

        ranker = FinalRanker()
        results = ranker.rank("Test", candidates, hr)

        c = results[0].components
        # Single candidate → norm_semantic = 1.0
        assert c["norm_semantic_score"] == pytest.approx(1.0, abs=1e-4)
        assert c["weighted_semantic"] == pytest.approx(0.40, abs=1e-4)
        assert c["weighted_availability"] == pytest.approx(0.20, abs=1e-4)
        assert c["weighted_evidence"] == pytest.approx(0.12, abs=1e-4)
        assert c["weighted_risk_penalty"] == pytest.approx(0.045, abs=1e-4)

        # raw_composite = sum of weighted - risk
        expected_raw = 0.40 + 0.20 + 0.12 - 0.045
        assert c["raw_composite"] == pytest.approx(expected_raw, abs=1e-4)


# ═══════════════════════════════════════════════════════════════════════════
# Telemetry Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTelemetry:
    """Tests for telemetry recording in RankedCandidate."""

    @patch("final_ranker.CrossEncoder")
    def test_telemetry_has_ranking_timestamp(
        self, mock_ce_cls: MagicMock
    ) -> None:
        """Telemetry includes a ranking_timestamp."""
        mock_model = _mock_cross_encoder_factory([5.0])
        mock_ce_cls.return_value = mock_model

        candidates = [_make_candidate("C001")]
        hr = {"C001": _make_hireability_result("C001")}

        ranker = FinalRanker()
        results = ranker.rank("Test", candidates, hr)

        assert "ranking_timestamp" in results[0].telemetry

    @patch("final_ranker.CrossEncoder")
    def test_telemetry_empty_defaults_when_hr_present(
        self, mock_ce_cls: MagicMock
    ) -> None:
        """No defaults are applied when HireabilityResult is present."""
        mock_model = _mock_cross_encoder_factory([5.0])
        mock_ce_cls.return_value = mock_model

        candidates = [_make_candidate("C001")]
        hr = {"C001": _make_hireability_result("C001")}

        ranker = FinalRanker()
        results = ranker.rank("Test", candidates, hr)

        assert results[0].telemetry["defaults_applied"] == []
        assert results[0].telemetry["warnings"] == []
