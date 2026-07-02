"""
Tests for Stage 3 — ReasoningEngine
=====================================
Team:    AlgoRhythms
Student: THEJAS J

Coverage:
    1.  ReasoningConfig validation (all fields)
    2.  ReasoningEngine initialization (valid / invalid config)
    3.  Input validation (empty JD, empty candidates)
    4.  Prompt construction (template filling, JD truncation, Phi-3 format)
    5.  LLM output parsing (valid JSON, regex fallback, missing keys)
    6.  Per-candidate error isolation (single failure, batch continues)
    7.  Fallback result correctness
    8.  ReasoningResult immutability (frozen dataclass)
    9.  to_dict() serialization
   10.  Grammar string validity
   11.  End-to-end batch processing with mocked Llama
   12.  Telemetry and warning propagation
   13.  Edge cases (minimal components, missing telemetry warnings)

All tests mock the Llama class so no .gguf model file is required.
The test suite runs in < 1 second.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from final_ranker import RankedCandidate
from reasoning_engine import (
    ReasoningConfig,
    ReasoningEngine,
    ReasoningResult,
    _FALLBACK_RISK,
    _FALLBACK_WHY,
    _JUSTIFICATION_GBNF,
    _SYSTEM_PROMPT,
    _USER_PROMPT_TEMPLATE,
)


# ═══════════════════════════════════════════════════════════════════════════
# Test Fixtures & Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_ranked_candidate(
    candidate_id: str = "C001",
    final_score: float = 0.85,
    norm_semantic_score: float = 0.92,
    availability_score: float = 0.80,
    evidence_coverage_score: float = 0.70,
    risk_score: float = 0.15,
    components: Dict[str, Any] | None = None,
    telemetry: Dict[str, Any] | None = None,
) -> RankedCandidate:
    """Build a RankedCandidate with controllable fields."""
    if components is None:
        components = {
            "raw_semantic_score": 7.234,
            "norm_semantic_score": norm_semantic_score,
            "weighted_semantic": 0.368,
            "weighted_availability": 0.200,
            "weighted_evidence": 0.140,
            "weighted_risk_penalty": 0.030,
            "raw_composite": 0.678,
            "was_clamped": False,
        }
    if telemetry is None:
        telemetry = {
            "defaults_applied": [],
            "warnings": [],
            "ranking_timestamp": "2026-06-09T12:00:00+00:00",
        }

    return RankedCandidate(
        candidate_id=candidate_id,
        final_score=final_score,
        norm_semantic_score=norm_semantic_score,
        availability_score=availability_score,
        evidence_coverage_score=evidence_coverage_score,
        risk_score=risk_score,
        components=components,
        telemetry=telemetry,
    )


def _make_valid_llm_response(
    why: str = "Strong semantic alignment with a score of 0.92 indicating excellent fit.",
    risk: str = "Minimal risk identified overall.",
    prompt_tokens: int = 200,
    completion_tokens: int = 50,
) -> Dict[str, Any]:
    """Build a mock Llama response dict with valid JSON output."""
    output_json = json.dumps({
        "why_selected": why,
        "risk_factors": risk,
    })
    return {
        "choices": [{"text": output_json}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


def _make_mock_llama(responses: List[Dict[str, Any]] | None = None):
    """Create a mock Llama instance that returns canned responses.

    Args:
        responses: List of response dicts. If None, returns a single
                   valid response for every call.

    Returns:
        MagicMock configured as a Llama instance.
    """
    mock = MagicMock()
    if responses is None:
        mock.return_value = _make_valid_llm_response()
        mock.side_effect = None
    else:
        mock.side_effect = responses
    return mock


# Path constants for patching
_LLAMA_CLASS_PATH = "reasoning_engine.Llama"
_GRAMMAR_CLASS_PATH = "reasoning_engine.LlamaGrammar"
_PATH_EXISTS_PATH = "reasoning_engine.Path.exists"
_PATH_IS_FILE_PATH = "reasoning_engine.Path.is_file"
_LLAMA_AVAILABLE_PATH = "reasoning_engine._LLAMA_CPP_AVAILABLE"


def _create_engine_with_mocks(
    config: ReasoningConfig | None = None,
) -> ReasoningEngine:
    """Create a ReasoningEngine with file-system checks bypassed.

    Uses patching to skip model_path validation during __init__
    and to bypass the llama-cpp-python availability check.
    """
    if config is None:
        config = ReasoningConfig(model_path="/fake/model.gguf")

    with patch(_PATH_EXISTS_PATH, return_value=True), \
         patch(_PATH_IS_FILE_PATH, return_value=True), \
         patch(_LLAMA_AVAILABLE_PATH, True):
        return ReasoningEngine(config=config)


# ═══════════════════════════════════════════════════════════════════════════
# ReasoningConfig Validation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestReasoningConfig:
    """Tests for ReasoningConfig validation."""

    def test_valid_config_passes(self) -> None:
        """A config with a valid (existing) model path passes."""
        with patch(_PATH_EXISTS_PATH, return_value=True), \
             patch(_PATH_IS_FILE_PATH, return_value=True):
            config = ReasoningConfig(model_path="/valid/model.gguf")
            errors = config.validate()
            assert errors == []

    def test_empty_model_path_rejected(self) -> None:
        """Empty model_path produces a validation error."""
        config = ReasoningConfig(model_path="")
        errors = config.validate()
        assert any("model_path" in e for e in errors)

    def test_whitespace_model_path_rejected(self) -> None:
        """Whitespace-only model_path produces a validation error."""
        config = ReasoningConfig(model_path="   ")
        errors = config.validate()
        assert any("model_path" in e for e in errors)

    def test_nonexistent_model_path_rejected(self) -> None:
        """model_path pointing to a nonexistent file is rejected."""
        config = ReasoningConfig(
            model_path="/nonexistent/path/model.gguf"
        )
        errors = config.validate()
        assert any("does not exist" in e for e in errors)

    def test_negative_n_ctx_rejected(self) -> None:
        """Negative n_ctx produces a validation error."""
        with patch(_PATH_EXISTS_PATH, return_value=True), \
             patch(_PATH_IS_FILE_PATH, return_value=True):
            config = ReasoningConfig(
                model_path="/valid/model.gguf", n_ctx=-1
            )
            errors = config.validate()
            assert any("n_ctx" in e for e in errors)

    def test_zero_n_threads_rejected(self) -> None:
        """Zero n_threads produces a validation error."""
        with patch(_PATH_EXISTS_PATH, return_value=True), \
             patch(_PATH_IS_FILE_PATH, return_value=True):
            config = ReasoningConfig(
                model_path="/valid/model.gguf", n_threads=0
            )
            errors = config.validate()
            assert any("n_threads" in e for e in errors)

    def test_negative_max_tokens_rejected(self) -> None:
        """Negative max_tokens produces a validation error."""
        with patch(_PATH_EXISTS_PATH, return_value=True), \
             patch(_PATH_IS_FILE_PATH, return_value=True):
            config = ReasoningConfig(
                model_path="/valid/model.gguf", max_tokens=-10
            )
            errors = config.validate()
            assert any("max_tokens" in e for e in errors)

    def test_negative_temperature_rejected(self) -> None:
        """Negative temperature produces a validation error."""
        with patch(_PATH_EXISTS_PATH, return_value=True), \
             patch(_PATH_IS_FILE_PATH, return_value=True):
            config = ReasoningConfig(
                model_path="/valid/model.gguf", temperature=-0.5
            )
            errors = config.validate()
            assert any("temperature" in e for e in errors)

    def test_zero_temperature_accepted(self) -> None:
        """Temperature of 0.0 (greedy decoding) is valid."""
        with patch(_PATH_EXISTS_PATH, return_value=True), \
             patch(_PATH_IS_FILE_PATH, return_value=True):
            config = ReasoningConfig(
                model_path="/valid/model.gguf", temperature=0.0
            )
            errors = config.validate()
            assert not any("temperature" in e for e in errors)

    def test_invalid_top_p_zero_rejected(self) -> None:
        """top_p of 0.0 is rejected (must be > 0)."""
        with patch(_PATH_EXISTS_PATH, return_value=True), \
             patch(_PATH_IS_FILE_PATH, return_value=True):
            config = ReasoningConfig(
                model_path="/valid/model.gguf", top_p=0.0
            )
            errors = config.validate()
            assert any("top_p" in e for e in errors)

    def test_invalid_top_p_above_one_rejected(self) -> None:
        """top_p above 1.0 is rejected."""
        with patch(_PATH_EXISTS_PATH, return_value=True), \
             patch(_PATH_IS_FILE_PATH, return_value=True):
            config = ReasoningConfig(
                model_path="/valid/model.gguf", top_p=1.5
            )
            errors = config.validate()
            assert any("top_p" in e for e in errors)

    def test_top_p_exactly_one_accepted(self) -> None:
        """top_p of 1.0 is valid."""
        with patch(_PATH_EXISTS_PATH, return_value=True), \
             patch(_PATH_IS_FILE_PATH, return_value=True):
            config = ReasoningConfig(
                model_path="/valid/model.gguf", top_p=1.0
            )
            errors = config.validate()
            assert not any("top_p" in e for e in errors)

    def test_negative_jd_max_chars_rejected(self) -> None:
        """Negative jd_max_chars produces a validation error."""
        with patch(_PATH_EXISTS_PATH, return_value=True), \
             patch(_PATH_IS_FILE_PATH, return_value=True):
            config = ReasoningConfig(
                model_path="/valid/model.gguf", jd_max_chars=-100
            )
            errors = config.validate()
            assert any("jd_max_chars" in e for e in errors)

    def test_multiple_errors_reported(self) -> None:
        """Multiple validation failures are all reported."""
        with patch(_PATH_EXISTS_PATH, return_value=True), \
             patch(_PATH_IS_FILE_PATH, return_value=True):
            config = ReasoningConfig(
                model_path="/valid/model.gguf",
                n_ctx=-1,
                n_threads=0,
                temperature=-1.0,
            )
            errors = config.validate()
            assert len(errors) == 3

    def test_default_values_reasonable(self) -> None:
        """Default config values are within expected ranges."""
        config = ReasoningConfig(model_path="/fake/model.gguf")
        assert config.n_ctx == 4096
        assert config.n_threads == 6
        assert config.max_tokens == 256
        assert config.temperature == 0.0
        assert config.top_p == 0.8
        assert config.jd_max_chars == 400
        assert config.seed == 42


# ═══════════════════════════════════════════════════════════════════════════
# ReasoningEngine Initialization Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestReasoningEngineInit:
    """Tests for ReasoningEngine construction and config validation."""

    def test_valid_config_accepted(self) -> None:
        """ReasoningEngine can be constructed with valid config."""
        engine = _create_engine_with_mocks()
        assert engine._config.model_path == "/fake/model.gguf"

    def test_invalid_config_raises_valueerror(self) -> None:
        """ReasoningEngine rejects an invalid config with ValueError."""
        config = ReasoningConfig(model_path="")  # invalid
        with patch(_LLAMA_AVAILABLE_PATH, True):
            with pytest.raises(ValueError, match="Invalid ReasoningConfig"):
                ReasoningEngine(config=config)

    def test_model_not_loaded_at_init(self) -> None:
        """Model is NOT loaded during __init__ (lazy loading)."""
        engine = _create_engine_with_mocks()
        assert engine._model is None
        assert engine._grammar is None


# ═══════════════════════════════════════════════════════════════════════════
# Input Validation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestInputValidation:
    """Tests for generate_justifications() input validation."""

    def test_empty_jd_raises_valueerror(self) -> None:
        """Empty JD text raises ValueError."""
        engine = _create_engine_with_mocks()
        candidates = [_make_ranked_candidate()]
        with pytest.raises(ValueError, match="jd_text must be a non-empty"):
            engine.generate_justifications("", candidates)

    def test_whitespace_jd_raises_valueerror(self) -> None:
        """Whitespace-only JD text raises ValueError."""
        engine = _create_engine_with_mocks()
        candidates = [_make_ranked_candidate()]
        with pytest.raises(ValueError, match="jd_text must be a non-empty"):
            engine.generate_justifications("   \n\t  ", candidates)

    def test_empty_candidates_raises_valueerror(self) -> None:
        """Empty candidates list raises ValueError."""
        engine = _create_engine_with_mocks()
        with pytest.raises(
            ValueError, match="ranked_candidates list must not be empty"
        ):
            engine.generate_justifications("Some JD text", [])


# ═══════════════════════════════════════════════════════════════════════════
# Prompt Construction Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPromptConstruction:
    """Tests for the _build_prompt method."""

    def test_prompt_contains_phi3_format_tags(self) -> None:
        """Prompt includes Phi-3 chat format tags."""
        engine = _create_engine_with_mocks()
        candidate = _make_ranked_candidate(candidate_id="C042")

        prompt = engine._build_prompt(
            jd_truncated="Senior Python Developer",
            candidate=candidate,
            rank=1,
            total=10,
        )

        assert "<|system|>" in prompt
        assert "<|end|>" in prompt
        assert "<|user|>" in prompt
        assert "<|assistant|>" in prompt

    def test_prompt_contains_system_prompt(self) -> None:
        """System prompt text is embedded in the full prompt."""
        engine = _create_engine_with_mocks()
        candidate = _make_ranked_candidate()

        prompt = engine._build_prompt(
            jd_truncated="ML Engineer role",
            candidate=candidate,
            rank=1,
            total=5,
        )

        assert "recruitment analytics engine" in prompt
        assert "why_selected" in prompt
        assert "risk_factors" in prompt

    def test_prompt_contains_candidate_id(self) -> None:
        """Candidate ID is injected into the prompt."""
        engine = _create_engine_with_mocks()
        candidate = _make_ranked_candidate(candidate_id="CAND_XYZ")

        prompt = engine._build_prompt(
            jd_truncated="Data Engineer",
            candidate=candidate,
            rank=3,
            total=15,
        )

        assert "CAND_XYZ" in prompt

    def test_prompt_contains_rank_context(self) -> None:
        """Rank and total are injected into the prompt."""
        engine = _create_engine_with_mocks()
        candidate = _make_ranked_candidate()

        prompt = engine._build_prompt(
            jd_truncated="Backend Developer",
            candidate=candidate,
            rank=7,
            total=12,
        )

        assert "#7 of 12" in prompt

    def test_prompt_contains_all_scores(self) -> None:
        """All numeric scores from the RankedCandidate are in the prompt."""
        engine = _create_engine_with_mocks()
        candidate = _make_ranked_candidate(
            final_score=0.7823,
            norm_semantic_score=0.91,
            availability_score=0.65,
            evidence_coverage_score=0.80,
            risk_score=0.22,
        )

        prompt = engine._build_prompt(
            jd_truncated="Full Stack Developer",
            candidate=candidate,
            rank=1,
            total=1,
        )

        assert "0.7823" in prompt
        assert "0.91" in prompt
        assert "0.65" in prompt
        assert "0.8" in prompt
        assert "0.22" in prompt

    def test_prompt_contains_component_breakdown(self) -> None:
        """Weighted component values are injected from components dict."""
        engine = _create_engine_with_mocks()
        components = {
            "weighted_semantic": 0.368,
            "weighted_availability": 0.200,
            "weighted_evidence": 0.140,
            "weighted_risk_penalty": 0.030,
        }
        candidate = _make_ranked_candidate(components=components)

        prompt = engine._build_prompt(
            jd_truncated="DevOps Engineer",
            candidate=candidate,
            rank=1,
            total=1,
        )

        assert "0.368" in prompt
        assert "0.2" in prompt
        assert "0.14" in prompt
        assert "0.03" in prompt

    def test_prompt_contains_jd_text(self) -> None:
        """JD text is injected into the prompt."""
        engine = _create_engine_with_mocks()
        candidate = _make_ranked_candidate()
        jd = "We need a Senior Rust Developer with 5 years experience"

        prompt = engine._build_prompt(
            jd_truncated=jd,
            candidate=candidate,
            rank=1,
            total=1,
        )

        assert "Senior Rust Developer" in prompt
        assert "5 years experience" in prompt

    def test_prompt_with_telemetry_warnings(self) -> None:
        """Telemetry warnings from Stage 2 are included in the prompt."""
        engine = _create_engine_with_mocks()
        telemetry = {
            "defaults_applied": [],
            "warnings": [
                "No HireabilityResult found for candidate 'C001'; "
                "pessimistic defaults applied"
            ],
            "ranking_timestamp": "2026-06-09T12:00:00+00:00",
        }
        candidate = _make_ranked_candidate(telemetry=telemetry)

        prompt = engine._build_prompt(
            jd_truncated="Any role",
            candidate=candidate,
            rank=1,
            total=1,
        )

        assert "pessimistic defaults" in prompt

    def test_prompt_with_no_warnings_shows_none(self) -> None:
        """When telemetry has no warnings, prompt shows 'None'."""
        engine = _create_engine_with_mocks()
        telemetry = {
            "defaults_applied": [],
            "warnings": [],
            "ranking_timestamp": "2026-06-09T12:00:00+00:00",
        }
        candidate = _make_ranked_candidate(telemetry=telemetry)

        prompt = engine._build_prompt(
            jd_truncated="Any role",
            candidate=candidate,
            rank=1,
            total=1,
        )

        assert "Telemetry Warnings: None" in prompt

    def test_prompt_with_missing_components_uses_na(self) -> None:
        """Missing component keys default to 'N/A' in the prompt."""
        engine = _create_engine_with_mocks()
        candidate = _make_ranked_candidate(components={})

        prompt = engine._build_prompt(
            jd_truncated="Any role",
            candidate=candidate,
            rank=1,
            total=1,
        )

        assert "N/A" in prompt

    def test_prompt_with_none_telemetry_no_crash(self) -> None:
        """None telemetry dict does not crash prompt construction."""
        engine = _create_engine_with_mocks()
        # Create candidate with None-like empty telemetry
        candidate = _make_ranked_candidate(telemetry={})

        prompt = engine._build_prompt(
            jd_truncated="Any role",
            candidate=candidate,
            rank=1,
            total=1,
        )

        # Should not crash and should contain basic structure
        assert "<|system|>" in prompt
        assert "<|assistant|>" in prompt


# ═══════════════════════════════════════════════════════════════════════════
# JD Truncation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestJDTruncation:
    """Tests for JD text truncation logic."""

    def test_short_jd_not_truncated(self) -> None:
        """JD shorter than jd_max_chars is returned unchanged."""
        engine = _create_engine_with_mocks()
        jd = "Short JD"
        result = engine._truncate_jd(jd)
        assert result == jd

    def test_exact_length_not_truncated(self) -> None:
        """JD exactly at jd_max_chars is returned unchanged."""
        config = ReasoningConfig(
            model_path="/fake/model.gguf", jd_max_chars=10
        )
        engine = _create_engine_with_mocks(config=config)
        jd = "ExactlyTen"  # 10 chars
        result = engine._truncate_jd(jd)
        assert result == jd

    def test_long_jd_truncated_with_ellipsis(self) -> None:
        """JD exceeding jd_max_chars is truncated with '...' appended."""
        config = ReasoningConfig(
            model_path="/fake/model.gguf", jd_max_chars=10
        )
        engine = _create_engine_with_mocks(config=config)
        jd = "This is a very long job description text"
        result = engine._truncate_jd(jd)
        assert len(result) == 13  # 10 + len("...")
        assert result.endswith("...")
        assert result == "This is a ..."

    def test_default_400_char_truncation(self) -> None:
        """Default config truncates at 400 characters."""
        engine = _create_engine_with_mocks()
        jd = "x" * 1500
        result = engine._truncate_jd(jd)
        assert len(result) == 403  # 400 + "..."
        assert result.endswith("...")


# ═══════════════════════════════════════════════════════════════════════════
# LLM Output Parsing Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestParseLLMOutput:
    """Tests for _parse_llm_output static method."""

    def test_valid_json_parsed(self) -> None:
        """Valid JSON with both keys is parsed correctly."""
        raw = json.dumps({
            "why_selected": "Great candidate.",
            "risk_factors": "Minimal risk identified.",
        })
        result = ReasoningEngine._parse_llm_output(raw)
        assert result["why_selected"] == "Great candidate."
        assert result["risk_factors"] == "Minimal risk identified."

    def test_valid_json_with_whitespace(self) -> None:
        """JSON with leading/trailing whitespace is parsed."""
        raw = '  \n  {"why_selected": "Good fit.", "risk_factors": "None."}\n  '
        result = ReasoningEngine._parse_llm_output(raw)
        assert result["why_selected"] == "Good fit."

    def test_regex_fallback_strips_preamble(self) -> None:
        """Text with conversational preamble before JSON is handled."""
        raw = (
            'Here is the JSON:\n'
            '{"why_selected": "Strong match.", "risk_factors": "Low risk."}'
        )
        result = ReasoningEngine._parse_llm_output(raw)
        assert result["why_selected"] == "Strong match."
        assert result["risk_factors"] == "Low risk."

    def test_regex_fallback_strips_backticks(self) -> None:
        """JSON wrapped in markdown backticks is extracted."""
        raw = (
            '```json\n'
            '{"why_selected": "Excellent.", "risk_factors": "None."}\n'
            '```'
        )
        result = ReasoningEngine._parse_llm_output(raw)
        assert result["why_selected"] == "Excellent."

    def test_missing_why_selected_raises(self) -> None:
        """JSON missing 'why_selected' key raises ValueError."""
        raw = json.dumps({"risk_factors": "Some risk."})
        with pytest.raises(ValueError, match="why_selected"):
            ReasoningEngine._parse_llm_output(raw)

    def test_missing_risk_factors_raises(self) -> None:
        """JSON missing 'risk_factors' key raises ValueError."""
        raw = json.dumps({"why_selected": "Good."})
        with pytest.raises(ValueError, match="risk_factors"):
            ReasoningEngine._parse_llm_output(raw)

    def test_completely_invalid_output_raises(self) -> None:
        """Non-JSON output with no extractable object raises ValueError."""
        raw = "This is just plain text with no JSON at all."
        with pytest.raises(ValueError, match="Failed to parse"):
            ReasoningEngine._parse_llm_output(raw)

    def test_empty_string_raises(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="Failed to parse"):
            ReasoningEngine._parse_llm_output("")

    def test_extra_keys_tolerated(self) -> None:
        """JSON with extra keys beyond the required two is accepted."""
        raw = json.dumps({
            "why_selected": "Match.",
            "risk_factors": "Low.",
            "extra_field": "ignored",
        })
        result = ReasoningEngine._parse_llm_output(raw)
        assert result["why_selected"] == "Match."
        assert result["risk_factors"] == "Low."
        # Extra keys are NOT in the result (only the two required)
        assert "extra_field" not in result

    def test_non_string_values_coerced_to_string(self) -> None:
        """Non-string values for required keys are coerced to str."""
        raw = json.dumps({
            "why_selected": 42,
            "risk_factors": True,
        })
        result = ReasoningEngine._parse_llm_output(raw)
        assert result["why_selected"] == "42"
        assert result["risk_factors"] == "True"

    def test_escaped_characters_in_json(self) -> None:
        """JSON with escaped characters is parsed correctly."""
        raw = json.dumps({
            "why_selected": 'Strong match with "Python" skills.',
            "risk_factors": "Notice period:\n90 days.",
        })
        result = ReasoningEngine._parse_llm_output(raw)
        assert '"Python"' in result["why_selected"]
        assert "\n" in result["risk_factors"]


# ═══════════════════════════════════════════════════════════════════════════
# GBNF Grammar Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGBNFGrammar:
    """Tests for the GBNF grammar string."""

    def test_grammar_string_is_nonempty(self) -> None:
        """The GBNF grammar string is defined and non-empty."""
        assert _JUSTIFICATION_GBNF
        assert len(_JUSTIFICATION_GBNF) > 50

    def test_grammar_defines_root_rule(self) -> None:
        """Grammar has a root rule."""
        assert "root" in _JUSTIFICATION_GBNF

    def test_grammar_enforces_why_selected_key(self) -> None:
        """Grammar requires the 'why_selected' key."""
        assert "why_selected" in _JUSTIFICATION_GBNF

    def test_grammar_enforces_risk_factors_key(self) -> None:
        """Grammar requires the 'risk_factors' key."""
        assert "risk_factors" in _JUSTIFICATION_GBNF

    def test_grammar_defines_string_rule(self) -> None:
        """Grammar defines a string production rule."""
        assert "string" in _JUSTIFICATION_GBNF
        assert "characters" in _JUSTIFICATION_GBNF

    def test_grammar_defines_escape_rule(self) -> None:
        """Grammar handles escaped characters in strings."""
        assert "escape" in _JUSTIFICATION_GBNF


# ═══════════════════════════════════════════════════════════════════════════
# Error Resilience Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestErrorResilience:
    """Tests for per-candidate error isolation and fallback behavior."""

    @patch(_GRAMMAR_CLASS_PATH)
    @patch(_LLAMA_CLASS_PATH)
    def test_single_failure_does_not_crash_batch(
        self, mock_llama_cls: MagicMock, mock_grammar_cls: MagicMock
    ) -> None:
        """One candidate failing mid-batch does not prevent others."""
        # Set up mock: first call succeeds, second raises, third succeeds
        mock_model = MagicMock()
        mock_model.side_effect = [
            _make_valid_llm_response(
                why="First candidate is great with strong Python skills and solid experience.",
                risk="Low risk across all dimensions.",
            ),
            RuntimeError("LLM inference crashed"),
            _make_valid_llm_response(
                why="Third candidate is good with relevant backend experience and strong fundamentals.",
                risk="Minimal risk identified overall.",
            ),
        ]
        mock_llama_cls.return_value = mock_model

        engine = _create_engine_with_mocks()
        engine._model = mock_model
        engine._grammar = MagicMock()

        candidates = [
            _make_ranked_candidate(candidate_id="C001"),
            _make_ranked_candidate(candidate_id="C002"),
            _make_ranked_candidate(candidate_id="C003"),
        ]

        results = engine.generate_justifications(
            "Python Developer", candidates
        )

        # All 3 results returned
        assert len(results) == 3

        # First succeeded
        assert results[0].why_selected == "First candidate is great with strong Python skills and solid experience."
        assert results[0].telemetry["used_fallback"] is False

        # Second used fallback
        assert results[1].why_selected == _FALLBACK_WHY
        assert results[1].risk_factors == _FALLBACK_RISK
        assert results[1].telemetry["used_fallback"] is True
        assert len(results[1].telemetry["warnings"]) >= 1

        # Third succeeded
        assert results[2].why_selected == "Third candidate is good with relevant backend experience and strong fundamentals."
        assert results[2].telemetry["used_fallback"] is False

    @patch(_GRAMMAR_CLASS_PATH)
    @patch(_LLAMA_CLASS_PATH)
    def test_all_candidates_fail_returns_all_fallbacks(
        self, mock_llama_cls: MagicMock, mock_grammar_cls: MagicMock
    ) -> None:
        """If every candidate fails, all get fallback results."""
        mock_model = MagicMock()
        mock_model.side_effect = Exception("Total LLM failure")
        mock_llama_cls.return_value = mock_model

        engine = _create_engine_with_mocks()
        engine._model = mock_model
        engine._grammar = MagicMock()

        candidates = [
            _make_ranked_candidate(candidate_id="C001"),
            _make_ranked_candidate(candidate_id="C002"),
        ]

        results = engine.generate_justifications(
            "Any role", candidates
        )

        assert len(results) == 2
        for r in results:
            assert r.why_selected == _FALLBACK_WHY
            assert r.risk_factors == _FALLBACK_RISK
            assert r.telemetry["used_fallback"] is True

    @patch(_GRAMMAR_CLASS_PATH)
    @patch(_LLAMA_CLASS_PATH)
    def test_fallback_result_has_zero_tokens(
        self, mock_llama_cls: MagicMock, mock_grammar_cls: MagicMock
    ) -> None:
        """Fallback results report zero prompt and completion tokens."""
        mock_model = MagicMock()
        mock_model.side_effect = Exception("Boom")
        mock_llama_cls.return_value = mock_model

        engine = _create_engine_with_mocks()
        engine._model = mock_model
        engine._grammar = MagicMock()

        candidates = [_make_ranked_candidate(candidate_id="C001")]
        results = engine.generate_justifications("JD", candidates)

        assert results[0].prompt_tokens == 0
        assert results[0].completion_tokens == 0

    @patch(_GRAMMAR_CLASS_PATH)
    @patch(_LLAMA_CLASS_PATH)
    def test_fallback_preserves_candidate_id_and_rank(
        self, mock_llama_cls: MagicMock, mock_grammar_cls: MagicMock
    ) -> None:
        """Fallback results preserve correct candidate_id and rank."""
        mock_model = MagicMock()
        mock_model.side_effect = Exception("Inference failed")
        mock_llama_cls.return_value = mock_model

        engine = _create_engine_with_mocks()
        engine._model = mock_model
        engine._grammar = MagicMock()

        candidates = [
            _make_ranked_candidate(candidate_id="CAND_99"),
        ]
        results = engine.generate_justifications("JD text", candidates)

        assert results[0].candidate_id == "CAND_99"
        assert results[0].rank == 1

    @patch(_GRAMMAR_CLASS_PATH)
    @patch(_LLAMA_CLASS_PATH)
    def test_fallback_records_error_message_in_telemetry(
        self, mock_llama_cls: MagicMock, mock_grammar_cls: MagicMock
    ) -> None:
        """Fallback telemetry includes the error message."""
        mock_model = MagicMock()
        mock_model.side_effect = ValueError("Bad output format")
        mock_llama_cls.return_value = mock_model

        engine = _create_engine_with_mocks()
        engine._model = mock_model
        engine._grammar = MagicMock()

        candidates = [_make_ranked_candidate()]
        results = engine.generate_justifications("JD", candidates)

        warnings = results[0].telemetry["warnings"]
        assert len(warnings) == 1
        assert "Bad output format" in warnings[0]


# ═══════════════════════════════════════════════════════════════════════════
# End-to-End Batch Processing Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEndToEndBatch:
    """End-to-end tests with mocked Llama model."""

    @patch(_GRAMMAR_CLASS_PATH)
    @patch(_LLAMA_CLASS_PATH)
    def test_successful_batch_of_three(
        self, mock_llama_cls: MagicMock, mock_grammar_cls: MagicMock
    ) -> None:
        """Three candidates processed successfully in order."""
        responses = [
            _make_valid_llm_response(
                why="Strong Python skills with a semantic score of 0.92 indicating excellent alignment.",
                risk="Moderate concern due to 90-day notice period.",
                prompt_tokens=180,
                completion_tokens=45,
            ),
            _make_valid_llm_response(
                why="Good Spark experience with a semantic score of 0.85 and solid data background.",
                risk="Minimal risk identified overall.",
                prompt_tokens=190,
                completion_tokens=40,
            ),
            _make_valid_llm_response(
                why="Solid ML background with a semantic score of 0.78 and relevant research experience.",
                risk="Low response rate at 42 percent raises concern.",
                prompt_tokens=195,
                completion_tokens=50,
            ),
        ]

        mock_model = MagicMock()
        mock_model.side_effect = responses
        mock_llama_cls.return_value = mock_model

        engine = _create_engine_with_mocks()
        engine._model = mock_model
        engine._grammar = MagicMock()

        candidates = [
            _make_ranked_candidate(candidate_id="C001", final_score=0.92),
            _make_ranked_candidate(candidate_id="C002", final_score=0.85),
            _make_ranked_candidate(candidate_id="C003", final_score=0.78),
        ]

        results = engine.generate_justifications(
            "Senior Python Developer with Spark experience", candidates
        )

        assert len(results) == 3

        # Verify rank order
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[2].rank == 3

        # Verify candidate IDs preserved
        assert results[0].candidate_id == "C001"
        assert results[1].candidate_id == "C002"
        assert results[2].candidate_id == "C003"

        # Verify content
        assert "Python" in results[0].why_selected
        assert "Spark" in results[1].why_selected
        assert "ML background" in results[2].why_selected

        # Verify token counts
        assert results[0].prompt_tokens == 180
        assert results[0].completion_tokens == 45

        # Verify telemetry
        for r in results:
            assert r.telemetry["used_fallback"] is False
            assert r.generation_time_ms >= 0

    @patch(_GRAMMAR_CLASS_PATH)
    @patch(_LLAMA_CLASS_PATH)
    def test_single_candidate_batch(
        self, mock_llama_cls: MagicMock, mock_grammar_cls: MagicMock
    ) -> None:
        """Single candidate batch processes correctly."""
        mock_model = MagicMock()
        mock_model.return_value = _make_valid_llm_response(
            why="Sole candidate demonstrates an excellent fit with strong relevant experience and skills.",
            risk="Minimal risk identified overall.",
        )
        mock_llama_cls.return_value = mock_model

        engine = _create_engine_with_mocks()
        engine._model = mock_model
        engine._grammar = MagicMock()

        candidates = [_make_ranked_candidate(candidate_id="C_SOLO")]
        results = engine.generate_justifications("Data Engineer", candidates)

        assert len(results) == 1
        assert results[0].candidate_id == "C_SOLO"
        assert results[0].rank == 1
        assert "Sole candidate" in results[0].why_selected

    @patch(_GRAMMAR_CLASS_PATH)
    @patch(_LLAMA_CLASS_PATH)
    def test_model_called_with_grammar(
        self, mock_llama_cls: MagicMock, mock_grammar_cls: MagicMock
    ) -> None:
        """The Llama model is invoked with the grammar parameter."""
        mock_model = MagicMock()
        mock_model.return_value = _make_valid_llm_response()
        mock_llama_cls.return_value = mock_model

        mock_grammar_instance = MagicMock()
        mock_grammar_cls.from_string.return_value = mock_grammar_instance

        engine = _create_engine_with_mocks()
        engine._model = mock_model
        engine._grammar = mock_grammar_instance

        candidates = [_make_ranked_candidate()]
        engine.generate_justifications("JD text", candidates)

        # Verify the model was called
        mock_model.assert_called_once()

        # Verify grammar was passed
        call_kwargs = mock_model.call_args
        assert call_kwargs.kwargs.get("grammar") is mock_grammar_instance

    @patch(_GRAMMAR_CLASS_PATH)
    @patch(_LLAMA_CLASS_PATH)
    def test_model_called_with_correct_temperature(
        self, mock_llama_cls: MagicMock, mock_grammar_cls: MagicMock
    ) -> None:
        """The Llama model is invoked with configured temperature."""
        config = ReasoningConfig(
            model_path="/fake/model.gguf", temperature=0.3
        )
        mock_model = MagicMock()
        mock_model.return_value = _make_valid_llm_response()
        mock_llama_cls.return_value = mock_model

        engine = _create_engine_with_mocks(config)
        engine._model = mock_model
        engine._grammar = MagicMock()

        candidates = [_make_ranked_candidate()]
        engine.generate_justifications("JD", candidates)

        call_kwargs = mock_model.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.3


# ═══════════════════════════════════════════════════════════════════════════
# ReasoningResult Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestReasoningResult:
    """Tests for the ReasoningResult frozen dataclass."""

    def test_frozen_immutability(self) -> None:
        """ReasoningResult is frozen — attribute assignment raises."""
        result = ReasoningResult(
            candidate_id="C001",
            rank=1,
            why_selected="Good.",
            risk_factors="Low.",
            generation_time_ms=150.0,
            prompt_tokens=200,
            completion_tokens=50,
            telemetry={"used_fallback": False, "warnings": []},
        )
        with pytest.raises(AttributeError):
            result.candidate_id = "C999"  # type: ignore[misc]

    def test_to_dict_serialization(self) -> None:
        """to_dict() returns a correctly serialized dictionary."""
        result = ReasoningResult(
            candidate_id="C042",
            rank=3,
            why_selected="Strong semantic alignment.",
            risk_factors="90-day notice period.",
            generation_time_ms=6234.567,
            prompt_tokens=210,
            completion_tokens=55,
            telemetry={"used_fallback": False, "warnings": []},
        )
        d = result.to_dict()

        assert d["candidate_id"] == "C042"
        assert d["rank"] == 3
        assert d["why_selected"] == "Strong semantic alignment."
        assert d["risk_factors"] == "90-day notice period."
        assert d["generation_time_ms"] == 6234.57  # rounded to 2 dp
        assert d["prompt_tokens"] == 210
        assert d["completion_tokens"] == 55
        assert d["telemetry"]["used_fallback"] is False

    def test_to_dict_is_json_serializable(self) -> None:
        """to_dict() output can be passed to json.dumps without error."""
        result = ReasoningResult(
            candidate_id="C001",
            rank=1,
            why_selected="Match.",
            risk_factors="Risk.",
            generation_time_ms=100.0,
            prompt_tokens=100,
            completion_tokens=30,
            telemetry={"used_fallback": False, "warnings": []},
        )
        serialized = json.dumps(result.to_dict())
        assert isinstance(serialized, str)
        assert "C001" in serialized


# ═══════════════════════════════════════════════════════════════════════════
# System Prompt & Template Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPromptTemplates:
    """Tests for the module-level prompt constants."""

    def test_system_prompt_mentions_json(self) -> None:
        """System prompt instructs the model to output JSON."""
        assert "JSON" in _SYSTEM_PROMPT

    def test_system_prompt_mentions_both_keys(self) -> None:
        """System prompt references both required output keys."""
        assert "why_selected" in _SYSTEM_PROMPT
        assert "risk_factors" in _SYSTEM_PROMPT

    def test_system_prompt_mentions_minimal_risk(self) -> None:
        """System prompt instructs minimal risk phrasing for low scores."""
        assert "Minimal risk identified" in _SYSTEM_PROMPT

    def test_user_template_has_all_placeholders(self) -> None:
        """User prompt template contains all required placeholders."""
        required_placeholders = [
            "{jd_text}",
            "{candidate_id}",
            "{rank}",
            "{total}",
            "{final_score}",
            "{norm_semantic_score}",
            "{availability_score}",
            "{evidence_coverage_score}",
            "{risk_score}",
            "{weighted_semantic}",
            "{weighted_availability}",
            "{weighted_evidence}",
            "{weighted_risk_penalty}",
            "{warnings_summary}",
        ]
        for ph in required_placeholders:
            assert ph in _USER_PROMPT_TEMPLATE, (
                f"Missing placeholder: {ph}"
            )

    def test_fallback_strings_are_nonempty(self) -> None:
        """Fallback strings are defined and non-empty."""
        assert _FALLBACK_WHY
        assert _FALLBACK_RISK
        assert len(_FALLBACK_WHY) > 5
        assert len(_FALLBACK_RISK) > 5


# ═══════════════════════════════════════════════════════════════════════════
# Model Loading Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestModelLoading:
    """Tests for lazy model and grammar loading."""

    @patch(_GRAMMAR_CLASS_PATH)
    @patch(_LLAMA_CLASS_PATH)
    def test_model_loaded_on_first_generate(
        self, mock_llama_cls: MagicMock, mock_grammar_cls: MagicMock
    ) -> None:
        """Model is loaded lazily on first generate_justifications call."""
        mock_model = MagicMock()
        mock_model.return_value = _make_valid_llm_response()
        mock_llama_cls.return_value = mock_model

        engine = _create_engine_with_mocks()
        assert engine._model is None

        candidates = [_make_ranked_candidate()]

        with patch.object(engine, "_load_model") as mock_load:
            with patch.object(engine, "_load_grammar"):
                # Force model to be set so processing can continue
                def set_model():
                    engine._model = mock_model

                mock_load.side_effect = set_model
                engine._grammar = MagicMock()

                engine.generate_justifications("JD", candidates)
                mock_load.assert_called_once()

    def test_load_model_raises_on_missing_file(self) -> None:
        """_load_model raises FileNotFoundError for missing model file."""
        engine = _create_engine_with_mocks()
        engine._config = ReasoningConfig(
            model_path="/definitely/nonexistent/model.gguf"
        )
        # Reset to force reload
        engine._model = None

        with pytest.raises(FileNotFoundError, match="not found"):
            engine._load_model()

    @patch(_GRAMMAR_CLASS_PATH)
    def test_grammar_loaded_via_from_string(
        self, mock_grammar_cls: MagicMock
    ) -> None:
        """Grammar is loaded using LlamaGrammar.from_string()."""
        mock_grammar_instance = MagicMock()
        mock_grammar_cls.from_string.return_value = mock_grammar_instance

        engine = _create_engine_with_mocks()
        engine._grammar = None  # force reload
        engine._load_grammar()

        mock_grammar_cls.from_string.assert_called_once_with(
            _JUSTIFICATION_GBNF
        )
        assert engine._grammar is mock_grammar_instance


# ═══════════════════════════════════════════════════════════════════════════
# Edge Case Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Tests for boundary conditions and unusual inputs."""

    @patch(_GRAMMAR_CLASS_PATH)
    @patch(_LLAMA_CLASS_PATH)
    def test_candidate_with_zero_risk_score(
        self, mock_llama_cls: MagicMock, mock_grammar_cls: MagicMock
    ) -> None:
        """Candidate with risk_score 0.0 processes without error."""
        mock_model = MagicMock()
        mock_model.return_value = _make_valid_llm_response(
            why="Perfect candidate.", risk="Minimal risk identified."
        )
        mock_llama_cls.return_value = mock_model

        engine = _create_engine_with_mocks()
        engine._model = mock_model
        engine._grammar = MagicMock()

        candidates = [
            _make_ranked_candidate(
                candidate_id="PERFECT",
                risk_score=0.0,
                final_score=0.95,
            ),
        ]

        results = engine.generate_justifications("Any JD", candidates)
        assert len(results) == 1
        assert results[0].candidate_id == "PERFECT"

    @patch(_GRAMMAR_CLASS_PATH)
    @patch(_LLAMA_CLASS_PATH)
    def test_candidate_with_max_risk_score(
        self, mock_llama_cls: MagicMock, mock_grammar_cls: MagicMock
    ) -> None:
        """Candidate with risk_score 1.0 processes without error."""
        mock_model = MagicMock()
        mock_model.return_value = _make_valid_llm_response(
            why="Marginal semantic match but the candidate shows some relevant experience in the domain.",
            risk="Extreme risk across all dimensions with high flight probability.",
        )
        mock_llama_cls.return_value = mock_model

        engine = _create_engine_with_mocks()
        engine._model = mock_model
        engine._grammar = MagicMock()

        candidates = [
            _make_ranked_candidate(
                candidate_id="HIGH_RISK",
                risk_score=1.0,
                final_score=0.10,
            ),
        ]

        results = engine.generate_justifications("Any JD", candidates)
        assert len(results) == 1
        assert "Extreme risk" in results[0].risk_factors

    @patch(_GRAMMAR_CLASS_PATH)
    @patch(_LLAMA_CLASS_PATH)
    def test_response_missing_usage_field(
        self, mock_llama_cls: MagicMock, mock_grammar_cls: MagicMock
    ) -> None:
        """LLM response missing 'usage' field defaults to 0 tokens."""
        response = {
            "choices": [
                {
                    "text": json.dumps({
                        "why_selected": "This candidate demonstrates strong alignment with the role requirements and relevant skills.",
                        "risk_factors": "Minimal risk identified overall.",
                    })
                }
            ],
            # No "usage" key
        }
        mock_model = MagicMock()
        mock_model.return_value = response
        mock_llama_cls.return_value = mock_model

        engine = _create_engine_with_mocks()
        engine._model = mock_model
        engine._grammar = MagicMock()

        candidates = [_make_ranked_candidate()]
        results = engine.generate_justifications("JD", candidates)

        assert results[0].prompt_tokens == 0
        assert results[0].completion_tokens == 0

    @patch(_GRAMMAR_CLASS_PATH)
    @patch(_LLAMA_CLASS_PATH)
    def test_fifteen_candidates_batch(
        self, mock_llama_cls: MagicMock, mock_grammar_cls: MagicMock
    ) -> None:
        """Batch of 15 candidates (max expected) processes correctly."""
        mock_model = MagicMock()
        mock_model.return_value = _make_valid_llm_response(
            why="Good candidate with strong experience and skills that align well with the role.",
            risk="Low risk across all dimensions.",
        )
        mock_llama_cls.return_value = mock_model

        engine = _create_engine_with_mocks()
        engine._model = mock_model
        engine._grammar = MagicMock()

        candidates = [
            _make_ranked_candidate(
                candidate_id=f"C{i:03d}",
                final_score=round(1.0 - i * 0.05, 2),
            )
            for i in range(15)
        ]

        results = engine.generate_justifications(
            "Senior Engineer", candidates
        )

        assert len(results) == 15

        # Verify ranks are sequential 1..15
        for i, r in enumerate(results):
            assert r.rank == i + 1

        # Verify all candidate IDs are present
        result_ids = {r.candidate_id for r in results}
        expected_ids = {f"C{i:03d}" for i in range(15)}
        assert result_ids == expected_ids

    @patch(_GRAMMAR_CLASS_PATH)
    @patch(_LLAMA_CLASS_PATH)
    def test_very_long_jd_is_truncated_in_prompt(
        self, mock_llama_cls: MagicMock, mock_grammar_cls: MagicMock
    ) -> None:
        """A very long JD is truncated before being injected into prompt."""
        mock_model = MagicMock()
        mock_model.return_value = _make_valid_llm_response()
        mock_llama_cls.return_value = mock_model

        engine = _create_engine_with_mocks()
        engine._model = mock_model
        engine._grammar = MagicMock()

        long_jd = "x" * 5000
        candidates = [_make_ranked_candidate()]

        results = engine.generate_justifications(long_jd, candidates)
        assert len(results) == 1

        # Verify the prompt was built with truncated JD
        call_args = mock_model.call_args
        prompt = call_args.args[0] if call_args.args else call_args.kwargs.get("prompt", "")
        # The JD should be truncated to 400 chars + "..."
        assert "x" * 400 in prompt
        assert "x" * 401 not in prompt
