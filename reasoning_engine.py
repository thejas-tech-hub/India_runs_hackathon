"""
Stage 3 — ReasoningEngine
==========================
Team:    AlgoRhythms
Student: THEJAS J

Purpose:
    Take the Top N (e.g., Top 10–15) RankedCandidate objects from Stage 2.
    Use a local, CPU-bound LLM to generate a short, punchy justification
    for the recruiter explaining the rationale behind each candidate's
    ranking.

Constraints:
    - CPU only (n_gpu_layers=0)
    - Strictly offline via llama-cpp-python
    - Structured JSON output enforced by GBNF grammar
    - Per-candidate error isolation (one failure never crashes the batch)

Model:
    Phi-3-mini-4k-instruct (Q4_K_M quantization, ~2.4 GB)

Output Schema (per candidate):
    {
      "why_selected": "<1-2 sentence highlight>",
      "risk_factors": "<1-2 sentence risk explanation>"
    }
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from llama_cpp import Llama, LlamaGrammar

    _LLAMA_CPP_AVAILABLE: bool = True
except ImportError:
    _LLAMA_CPP_AVAILABLE = False
    Llama = None  # type: ignore[assignment,misc]
    LlamaGrammar = None  # type: ignore[assignment,misc]

from final_ranker import RankedCandidate

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# GBNF Grammar
# ═══════════════════════════════════════════════════════════════════════════

_JUSTIFICATION_GBNF: str = r"""
root   ::= "{" ws "\"why_selected\"" ws ":" ws string "," ws "\"risk_factors\"" ws ":" ws string ws "}"
string ::= "\"" characters "\""
characters ::= character*
character  ::= [^"\\] | "\\" escape
escape     ::= ["\\nrt/]
ws     ::= [ \t\n]*
""".strip()


# ═══════════════════════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT: str = (
    "You are a recruitment analytics engine. Your sole purpose is to "
    "produce a JSON object explaining why a candidate was ranked at "
    "their position.\n\n"
    "Rules:\n"
    "1. Output ONLY a valid JSON object. No markdown, no backticks, "
    "no explanation outside the JSON.\n"
    "2. The JSON must have exactly two keys: \"why_selected\" and "
    "\"risk_factors\".\n"
    "3. \"why_selected\": 1-2 sentences highlighting the candidate's "
    "semantic match strength and strongest behavioral/evidence signals.\n"
    "4. \"risk_factors\": 1-2 sentences identifying specific risks. "
    "If the risk score is below 0.10, state \"Minimal risk identified.\"\n"
    "5. Reference the actual numeric scores provided. Do not fabricate "
    "numbers.\n"
    "6. Be concise and recruiter-friendly. Avoid jargon.\n"
    "7. Never mention topics unrelated to recruitment.\n"
    "8. Never invent examples, stories, recipes, food, sports, or unrelated facts.\n"
    "9. If uncertain, explain only using the provided scores.\n"
    "10. Do not speculate beyond the supplied candidate data."
    )


# ═══════════════════════════════════════════════════════════════════════════
# User Prompt Template
# ═══════════════════════════════════════════════════════════════════════════

_USER_PROMPT_TEMPLATE: str = (
    "JOB DESCRIPTION:\n"
    "{jd_text}\n\n"
    "CANDIDATE RANKING DATA:\n"
    "- Candidate ID: {candidate_id}\n"
    "- Final Rank: #{rank} of {total}\n"
    "- Final Score: {final_score}\n"
    "- Semantic Match Score: {norm_semantic_score}\n"
    "- Availability Score: {availability_score}\n"
    "- Evidence Coverage Score: {evidence_coverage_score}\n"
    "- Risk Score: {risk_score}\n"
    "- Score Breakdown:\n"
    "  - Weighted Semantic: {weighted_semantic}\n"
    "  - Weighted Availability: {weighted_availability}\n"
    "  - Weighted Evidence: {weighted_evidence}\n"
    "  - Weighted Risk Penalty: {weighted_risk_penalty}\n"
    "- Telemetry Warnings: {warnings_summary}\n\n"
    "Generate the JSON justification."
)


# ═══════════════════════════════════════════════════════════════════════════
# Fallback Result
# ═══════════════════════════════════════════════════════════════════════════

_FALLBACK_WHY: str = "Justification generation failed."
_FALLBACK_RISK: str = "Unable to assess — manual review recommended."


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ReasoningConfig:
    """Configuration for ReasoningEngine.

    Attributes:
        model_path: Absolute path to the .gguf model file.
        n_ctx: Context window size in tokens.
        n_threads: Number of CPU threads for inference.
        max_tokens: Maximum output tokens per candidate generation.
        temperature: Sampling temperature (lower = more deterministic).
        top_p: Nucleus sampling threshold.
        jd_max_chars: Maximum characters to retain from the JD.
        seed: Random seed for reproducibility.
    """

    model_path: str = ""
    n_ctx: int = 4096
    n_threads: int = 6
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.8
    jd_max_chars: int = 400
    seed: int = 42

    def validate(self) -> List[str]:
        """Validate configuration integrity.

        Returns:
            List of human-readable error messages. Empty list = valid.
        """
        errors: List[str] = []

        # Model path must be a non-empty string
        if not self.model_path or not self.model_path.strip():
            errors.append("model_path must be a non-empty string")

        # Model path must point to an existing file
        if self.model_path and self.model_path.strip():
            model_file = Path(self.model_path)
            if not model_file.exists():
                errors.append(
                    f"model_path does not exist: {self.model_path}"
                )
            elif not model_file.is_file():
                errors.append(
                    f"model_path is not a file: {self.model_path}"
                )

        # Context window must be positive
        if self.n_ctx <= 0:
            errors.append(
                f"n_ctx must be positive, got {self.n_ctx}"
            )

        # Thread count must be positive
        if self.n_threads <= 0:
            errors.append(
                f"n_threads must be positive, got {self.n_threads}"
            )

        # Max tokens must be positive
        if self.max_tokens <= 0:
            errors.append(
                f"max_tokens must be positive, got {self.max_tokens}"
            )

        # Temperature must be non-negative
        if self.temperature < 0:
            errors.append(
                f"temperature must be non-negative, got {self.temperature}"
            )

        # top_p must be in (0, 1]
        if self.top_p <= 0 or self.top_p > 1.0:
            errors.append(
                f"top_p must be in (0.0, 1.0], got {self.top_p}"
            )

        # jd_max_chars must be positive
        if self.jd_max_chars <= 0:
            errors.append(
                f"jd_max_chars must be positive, got {self.jd_max_chars}"
            )

        return errors


# ═══════════════════════════════════════════════════════════════════════════
# Result
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ReasoningResult:
    """Immutable result for a single candidate's LLM-generated justification.

    Attributes:
        candidate_id: Unique candidate identifier.
        rank: 1-based rank position in the Stage 2 output.
        why_selected: LLM-generated highlight of the candidate's strengths.
        risk_factors: LLM-generated explanation of the candidate's risks.
        generation_time_ms: Wall-clock time for this candidate's generation.
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the LLM output.
        telemetry: Audit trail including fallback indicators and warnings.
    """

    candidate_id: str
    rank: int
    why_selected: str
    risk_factors: str
    generation_time_ms: float
    prompt_tokens: int
    completion_tokens: int
    telemetry: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for downstream consumption."""
        return {
            "candidate_id": self.candidate_id,
            "rank": self.rank,
            "why_selected": self.why_selected,
            "risk_factors": self.risk_factors,
            "generation_time_ms": round(self.generation_time_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "telemetry": self.telemetry,
        }


# ═══════════════════════════════════════════════════════════════════════════
# ReasoningEngine
# ═══════════════════════════════════════════════════════════════════════════


class ReasoningEngine:
    """Stage 3 — LLM-powered candidate justification generator.

    Takes the Top N RankedCandidate objects from Stage 2 and generates
    a concise, recruiter-friendly JSON justification for each candidate
    using a local, CPU-bound LLM via llama-cpp-python.

    The output is constrained by a GBNF grammar to guarantee valid,
    parseable JSON with exactly two keys: ``why_selected`` and
    ``risk_factors``.

    Usage::

        config = ReasoningConfig(model_path="/path/to/phi3.gguf")
        engine = ReasoningEngine(config)
        results = engine.generate_justifications(jd_text, ranked_candidates)
        for r in results:
            print(r.candidate_id, r.why_selected)

    Parameters
    ----------
    config : ReasoningConfig
        Configuration object specifying model path, inference parameters,
        and prompt template limits.
    """

    def __init__(self, config: ReasoningConfig) -> None:
        """Initialize ReasoningEngine with configuration.

        Args:
            config: ReasoningConfig instance.

        Raises:
            ImportError: If llama-cpp-python is not installed.
            ValueError: If config fails validation.
            FileNotFoundError: If model_path does not exist.
        """
        if not _LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is required for ReasoningEngine. "
                "Install it with: pip install llama-cpp-python"
            )

        self._config: ReasoningConfig = config
        errors: List[str] = self._config.validate()
        if errors:
            raise ValueError(
                f"Invalid ReasoningConfig: {'; '.join(errors)}"
            )

        # Lazy-loaded model and grammar — initialised on first call
        self._model: Optional[Llama] = None
        self._grammar: Optional[LlamaGrammar] = None

    # ── Public API ───────────────────────────────────────────────────────

    def generate_justifications(
        self,
        jd_text: str,
        ranked_candidates: List[RankedCandidate],
    ) -> List[ReasoningResult]:
        """Generate LLM justifications for each ranked candidate.

        Parameters
        ----------
        jd_text : str
            The job description text (will be truncated to jd_max_chars).
        ranked_candidates : list[RankedCandidate]
            Top N candidates from Stage 2, sorted descending by final_score.

        Returns
        -------
        list[ReasoningResult]
            One ReasoningResult per candidate, in the same rank order.
            Failed candidates receive fallback text (never raises).

        Raises
        ------
        ValueError
            If ``jd_text`` is empty or ``ranked_candidates`` is empty.
        """
        if not jd_text or not jd_text.strip():
            raise ValueError("jd_text must be a non-empty string")
        if not ranked_candidates:
            raise ValueError("ranked_candidates list must not be empty")

        n_total: int = len(ranked_candidates)
        logger.info(
            "Stage 3: Generating justifications for %d candidates",
            n_total,
        )

        # Ensure model and grammar are loaded
        self._load_model()
        self._load_grammar()

        # Truncate JD once for all candidates
        jd_truncated: str = self._truncate_jd(jd_text)

        results: List[ReasoningResult] = []
        batch_start: float = time.perf_counter()

        for idx, candidate in enumerate(ranked_candidates):
            rank: int = idx + 1
            result: ReasoningResult = self._process_single_candidate(
                jd_truncated=jd_truncated,
                candidate=candidate,
                rank=rank,
                total=n_total,
            )
            results.append(result)
            logger.info(
                "Stage 3: [%d/%d] %s — %.0f ms",
                rank,
                n_total,
                candidate.candidate_id,
                result.generation_time_ms,
            )

        batch_elapsed: float = (
            (time.perf_counter() - batch_start) * 1000.0
        )
        logger.info(
            "Stage 3: Batch complete. %d candidates in %.1f ms",
            n_total,
            batch_elapsed,
        )

        return results

    # ── Private: Model and Grammar Loading ───────────────────────────────

    def _load_model(self) -> None:
        """Load or return cached Llama model (CPU only).

        Raises
        ------
        FileNotFoundError
            If the model file does not exist at the configured path.
        RuntimeError
            If llama.cpp fails to load the model (corrupt GGUF, etc.).
        """
        if self._model is not None:
            return

        model_path: Path = Path(self._config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self._config.model_path}"
            )

        logger.info(
            "Loading LLM: %s (n_ctx=%d, n_threads=%d, device=cpu)",
            self._config.model_path,
            self._config.n_ctx,
            self._config.n_threads,
        )

        try:
            self._model = Llama(
                model_path=str(model_path),
                n_ctx=self._config.n_ctx,
                n_threads=self._config.n_threads,
                n_gpu_layers=0,
                seed=self._config.seed,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load LLM from {self._config.model_path}: "
                f"{exc}"
            ) from exc

    def _load_grammar(self) -> None:
        """Load or return cached GBNF grammar for structured output."""
        if self._grammar is not None:
            return

        logger.info("Compiling GBNF grammar for JSON output constraint")
        self._grammar = LlamaGrammar.from_string(_JUSTIFICATION_GBNF)

    # ── Private: Per-Candidate Processing ────────────────────────────────

    def _process_single_candidate(
        self,
        jd_truncated: str,
        candidate: RankedCandidate,
        rank: int,
        total: int,
    ) -> ReasoningResult:
        """Process a single candidate with full error isolation.

        A failure for this candidate will produce a fallback
        ReasoningResult and log a warning — it will never propagate
        to crash the batch.

        Parameters
        ----------
        jd_truncated : str
            The truncated job description text.
        candidate : RankedCandidate
            The candidate to generate justification for.
        rank : int
            1-based rank position.
        total : int
            Total number of candidates being processed.

        Returns
        -------
        ReasoningResult
            Either a successful LLM-generated result or a fallback.
        """
        telemetry: Dict[str, Any] = {
            "used_fallback": False,
            "output_repaired": False,
            "warnings": [],
        }

        start_time: float = time.perf_counter()

        try:
            # Build prompt
            prompt: str = self._build_prompt(
                jd_truncated, candidate, rank, total
            )

            # Run inference with grammar constraint
            assert self._model is not None  # guaranteed by caller
            assert self._grammar is not None

            response = self._model(
                prompt,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                top_p=self._config.top_p,
                grammar=self._grammar,
                echo=False,
            )

            # Extract raw output text
            raw_output: str = response["choices"][0]["text"]

            # Extract token usage
            usage: Dict[str, int] = response.get("usage", {})
            prompt_tokens: int = usage.get("prompt_tokens", 0)
            completion_tokens: int = usage.get("completion_tokens", 0)

            # Parse structured output
            parsed: Dict[str, str] = self._parse_llm_output(raw_output)

            # ── Repair before validation ──────────────────────────
            warnings_before: int = len(telemetry["warnings"])
            why_text: str = self._repair_output(
                parsed["why_selected"],
                candidate.candidate_id,
                telemetry["warnings"],
            )
            risk_text: str = self._repair_output(
                parsed["risk_factors"],
                candidate.candidate_id,
                telemetry["warnings"],
            )
            if len(telemetry["warnings"]) > warnings_before:
                telemetry["output_repaired"] = True

            # ── Post-parse validation ────────────────────────────
            combined_text: str = (why_text + " " + risk_text).lower()

            # Gate 1: Hallucinated content (existing banned words)
            banned_words = [
                "cookie",
                "butter",
                "sugar",
                "chocolate",
                "recipe",
                "cake",
            ]

            if any(word in combined_text for word in banned_words):
                raise ValueError(
                    "Hallucinated content detected"
                )

            # Gate 2: Corrupted token detection
            # Catches "0ayer", "0ayer_0053591", and similar
            # malformed numeric/text fragments from LLM decode
            # errors.
            corrupted_patterns = [
                r"0ayer",           # known corruption token
                r"\d[a-z]{2,}\d",   # e.g. "0ayer0", "3xyz5"
                r"[a-z]\d{4,}",     # e.g. "a00535", ID-like leak
            ]

            for pattern in corrupted_patterns:
                if re.search(pattern, combined_text):
                    raise ValueError("Corrupted token detected")

            # Gate 3: Candidate ID leakage detection
            cid_lower: str = candidate.candidate_id.lower()
            if cid_lower in combined_text:
                raise ValueError(
                    "Candidate ID leaked into explanation"
                )

            # Gate 4: Minimum quality validation
            why_word_count: int = len(why_text.split())
            risk_word_count: int = len(risk_text.split())

            if why_word_count < 8:
                raise ValueError(
                    f"why_selected too short: {why_word_count} words "
                    f"(minimum 8 required)"
                )

            if risk_word_count < 3:
                raise ValueError(
                    f"risk_factors too short: {risk_word_count} words "
                    f"(minimum 3 required)"
                )

            # Gate 5: Sentence-completion validation
            valid_terminators = (".", "!", "?")

            if not why_text.rstrip().endswith(valid_terminators):
                raise ValueError(
                    "Incomplete sentence detected"
                )

            if not risk_text.rstrip().endswith(valid_terminators):
                raise ValueError(
                    "Incomplete sentence detected"
                )

            elapsed_ms: float = (
                (time.perf_counter() - start_time) * 1000.0
            )

            return ReasoningResult(
                candidate_id=candidate.candidate_id,
                rank=rank,
                why_selected=why_text,
                risk_factors=risk_text,
                generation_time_ms=round(elapsed_ms, 2),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                telemetry=telemetry,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.warning(
                "Stage 3: Candidate '%s' failed — using fallback. "
                "Error: %s",
                candidate.candidate_id,
                str(exc),
            )
            telemetry["used_fallback"] = True
            telemetry["warnings"].append(
                f"LLM generation failed: {str(exc)}"
            )

            return ReasoningResult(
                candidate_id=candidate.candidate_id,
                rank=rank,
                why_selected=_FALLBACK_WHY,
                risk_factors=_FALLBACK_RISK,
                generation_time_ms=round(elapsed_ms, 2),
                prompt_tokens=0,
                completion_tokens=0,
                telemetry=telemetry,
            )

    # ── Private: Prompt Construction ─────────────────────────────────────

    def _build_prompt(
        self,
        jd_truncated: str,
        candidate: RankedCandidate,
        rank: int,
        total: int,
    ) -> str:
        """Build the full prompt in Phi-3 chat format.

        Format:
            <|system|>{system_prompt}<|end|>
            <|user|>{user_prompt}<|end|>
            <|assistant|>

        Parameters
        ----------
        jd_truncated : str
            The truncated job description text.
        candidate : RankedCandidate
            The candidate to justify.
        rank : int
            1-based rank position.
        total : int
            Total number of candidates.

        Returns
        -------
        str
            The fully formatted prompt string.
        """
        # Extract component breakdown values safely
        components: Dict[str, Any] = candidate.components or {}
        weighted_semantic: str = str(
            components.get("weighted_semantic", "N/A")
        )
        weighted_availability: str = str(
            components.get("weighted_availability", "N/A")
        )
        weighted_evidence: str = str(
            components.get("weighted_evidence", "N/A")
        )
        weighted_risk_penalty: str = str(
            components.get("weighted_risk_penalty", "N/A")
        )

        # Extract telemetry warnings
        candidate_telemetry: Dict[str, Any] = candidate.telemetry or {}
        warnings: List[str] = candidate_telemetry.get("warnings", [])
        warnings_summary: str = (
            "; ".join(warnings) if warnings else "None"
        )

        # Fill user prompt template
        user_prompt: str = _USER_PROMPT_TEMPLATE.format(
            jd_text=jd_truncated,
            candidate_id=candidate.candidate_id,
            rank=rank,
            total=total,
            final_score=candidate.final_score,
            norm_semantic_score=candidate.norm_semantic_score,
            availability_score=candidate.availability_score,
            evidence_coverage_score=candidate.evidence_coverage_score,
            risk_score=candidate.risk_score,
            weighted_semantic=weighted_semantic,
            weighted_availability=weighted_availability,
            weighted_evidence=weighted_evidence,
            weighted_risk_penalty=weighted_risk_penalty,
            warnings_summary=warnings_summary,
        )

        # Assemble in Phi-3 chat format
        prompt: str = (
            f"<|system|>\n{_SYSTEM_PROMPT}<|end|>\n"
            f"<|user|>\n{user_prompt}<|end|>\n"
            f"<|assistant|>\n"
        )

        return prompt

    # ── Private: Output Parsing ──────────────────────────────────────────

    @staticmethod
    def _parse_llm_output(raw: str) -> Dict[str, str]:
        """Parse and validate LLM output with fallback regex cleanup.

        With GBNF grammar enforcement, the output should always be
        valid JSON. This method adds a defense-in-depth regex fallback
        for edge cases.

        Parameters
        ----------
        raw : str
            Raw text output from the LLM.

        Returns
        -------
        dict[str, str]
            Dictionary with exactly ``why_selected`` and ``risk_factors``.

        Raises
        ------
        ValueError
            If the output cannot be parsed or is missing required keys.
        """
        cleaned: str = raw.strip()

        # Attempt direct JSON parse (should always work with GBNF)
        result: Optional[Dict[str, Any]] = None

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: extract first JSON object via regex
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        if result is None:
            raise ValueError(
                f"Failed to parse LLM output as JSON: "
                f"{cleaned[:200]!r}"
            )

        # Validate required keys
        if "why_selected" not in result:
            raise ValueError(
                "LLM output missing required key: 'why_selected'"
            )
        if "risk_factors" not in result:
            raise ValueError(
                "LLM output missing required key: 'risk_factors'"
            )

        return {
            "why_selected": str(result["why_selected"]),
            "risk_factors": str(result["risk_factors"]),
        }

    # ── Private: Output Repair ──────────────────────────────────────────

    @staticmethod
    def _repair_output(
        text: str,
        candidate_id: str,
        telemetry_warnings: list,
    ) -> str:
        """Attempt to repair known LLM output corruptions.

        Applies deterministic repair rules to fix common quantization
        artifacts and formatting glitches before validation gates run.
        All repairs are logged to telemetry for auditability.

        Strategy: **trim** corrupted fragments rather than substituting
        generic phrases, so the surrounding sentence remains natural.

        Parameters
        ----------
        text : str
            The raw text field (why_selected or risk_factors).
        candidate_id : str
            The candidate ID to detect and remove if leaked.
        telemetry_warnings : list
            Mutable list to append repair log entries to.

        Returns
        -------
        str
            The repaired text.
        """
        original: str = text

        # ── Repair 1: Fix corrupted prefixes on legitimate words ─
        # Q4_K_M sometimes prepends a stray "0" to real words.
        # These have clear correct forms, so direct substitution
        # is safe and produces natural text.
        prefix_fixes: dict = {
            "0player": "player",
            "0ployer": "employer",
            "0mployer": "employer",
        }
        for corrupt, fix in prefix_fixes.items():
            if corrupt in text.lower():
                text = re.sub(
                    re.escape(corrupt),
                    fix,
                    text,
                    flags=re.IGNORECASE,
                )

        # ── Repair 2: Trim corrupted score tokens ────────────────
        # "0ayer" is a Q4_K_M artifact where the model tried to
        # emit a numeric score (e.g. "0.7671") but produced garbled
        # bytes.  These tokens appear after prepositions like "at"
        # or "of" (e.g. "high at 0ayer", "score of 0ayer_0053591").
        #
        # Instead of substituting a phrase (which creates awkward
        # output like "high at strong alignment with requirements"),
        # REMOVE the preposition + corruption as a unit, leaving
        # the sentence grammatically intact.
        #
        # "...high at 0ayer, suggesting..." → "...high, suggesting..."
        # "...score of 0ayer_0053591."      → "...score."

        # 2a: Remove "at/of" + corruption + optional trailing junk
        text = re.sub(
            r"\s+(?:at|of)\s+0[a-z]*ayer[\w_]*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # 2b: Fallback — standalone corruption not after preposition
        text = re.sub(
            r"0[a-z]*ayer[\w_]*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # ── Repair 3: Trim legacy "layer" repair remnants ────────
        # Previous repair versions mapped "0ayer" → "layer" or
        # "0ayer" → "strong alignment with requirements", creating
        # artifacts.  Trim these while protecting legitimate words
        # (player, employer, multilayer, layered).

        # 3a: Remove "at/of" + "layer_NNNNN" (ID-like remnants)
        text = re.sub(
            r"\s+(?:at|of)\s+layer_\d+",
            "",
            text,
        )

        # 3b: Remove standalone "layer_NNNNN"
        text = re.sub(r"\blayer_\d+", "", text)

        # 3c: Remove "at/of" + isolated "layer" (not part of a word)
        # The negative lookahead protects "layered", "layers", etc.
        text = re.sub(
            r"\s+(?:at|of)\s+layer(?![a-zA-Z_\d])",
            "",
            text,
        )

        # 3d: Remove "at/of" + "strong alignment with requirements"
        # remnant from the intermediate repair version, with
        # optional trailing _NNNNN from ID leakage.
        text = re.sub(
            r"\s+(?:at|of)\s+strong alignment with requirements"
            r"(?:[_]\d+)*",
            "",
            text,
        )

        # 3e: Remove standalone "strong alignment with requirements"
        # followed by _NNNNN (ID leak remnant only — do NOT remove
        # the phrase when it stands alone as valid text).
        text = re.sub(
            r"strong alignment with requirements[_]\d+",
            "",
            text,
        )

        # ── Repair 4: Remove candidate ID leakage ────────────────
        if candidate_id.lower() in text.lower():
            text = re.sub(
                re.escape(candidate_id),
                "the candidate",
                text,
                flags=re.IGNORECASE,
            )

        # ── Repair 5: Fix letter-digit score concatenation ───────
        # e.g. "score0.4375" → "score 0.4375"
        text = re.sub(
            r"([a-zA-Z])(\d+\.\d+)",
            r"\1 \2",
            text,
        )

        # ── Repair 6: Clean punctuation / whitespace artifacts ───
        # Trimming can leave orphaned punctuation or extra spaces.
        text = re.sub(r"\s+([,.])", r"\1", text)   # " ," → ","
        text = re.sub(r"  +", " ", text)            # collapse spaces
        text = text.strip()

        # ── Repair 7: Ensure sentence termination ────────────────
        stripped: str = text.rstrip()
        if stripped and not stripped.endswith((".", "!", "?")):
            text = stripped + "."

        # ── Log repairs ──────────────────────────────────────────
        if text != original:
            telemetry_warnings.append(
                f"Output repaired: '{original[:80]}...' → "
                f"'{text[:80]}...'"
            )

        return text

    # ── Private: Utility ─────────────────────────────────────────────────

    def _truncate_jd(self, jd_text: str) -> str:
        """Truncate job description text to configured maximum length.

        Truncation is character-based. If the text exceeds the limit,
        it is cut at the boundary and ``...`` is appended.

        Parameters
        ----------
        jd_text : str
            The full job description text.

        Returns
        -------
        str
            The truncated (or original) JD text.
        """
        max_chars: int = self._config.jd_max_chars
        if len(jd_text) <= max_chars:
            return jd_text
        return jd_text[:max_chars] + "..."
