"""
Stage 2 — FinalRanker
======================
Team:    AlgoRhythms
Student: THEJAS J

Purpose:
    Take the Top K candidates returned by Stage 1 (HybridRetriever),
    compute their deep semantic relevance via a Cross-Encoder, merge
    it with their behavioral intelligence profiles from Stage 0B
    (HireabilityEvaluator), and produce the absolute, final sorted
    ranking of candidates.

Constraints:
    - CPU only (device='cpu')
    - Target runtime < 15 seconds for up to 500 candidates
    - Strict type hinting and dataclass configuration
    - No BM25, FAISS, or bi-encoder usage (Stage 1 only)

Cross-Encoder Model:
    cross-encoder/ms-marco-MiniLM-L-6-v2 (22M params, MS MARCO trained)

Fusion Formula:
    Final_Score = (w_semantic × norm_semantic_score)
               + (w_availability × availability_score)
               + (w_evidence × evidence_coverage_score)
               - (w_risk × risk_score)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

from hireability_evaluator import HireabilityResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULT_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_NORM_EPSILON: float = 1e-9


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class RankerConfig:
    """Configuration for FinalRanker composite scoring.

    All weights are non-negative. The additive weights
    (w_semantic + w_availability + w_evidence) must not exceed 1.0.
    The risk penalty weight (w_risk) must not exceed 1.0.

    Attributes:
        w_semantic: Weight for the normalized cross-encoder score.
        w_availability: Weight for the Stage 0B availability score.
        w_evidence: Weight for the Stage 0B evidence coverage score.
        w_risk: Weight for the Stage 0B risk penalty (subtracted).
        model_name: HuggingFace model identifier for the cross-encoder.
    """

    w_semantic: float = 0.40
    w_availability: float = 0.25
    w_evidence: float = 0.20
    w_risk: float = 0.15
    model_name: str = _DEFAULT_MODEL_NAME

    def validate(self) -> List[str]:
        """Validate configuration integrity.

        Returns:
            List of human-readable error messages. Empty list = valid.
        """
        errors: List[str] = []

        # All weights must be non-negative
        for attr_name in (
            "w_semantic",
            "w_availability",
            "w_evidence",
            "w_risk",
        ):
            val: float = getattr(self, attr_name)
            if val < 0:
                errors.append(
                    f"{attr_name} must be non-negative, got {val}"
                )

        # Additive weights must not exceed 1.0
        additive_sum: float = (
            self.w_semantic + self.w_availability + self.w_evidence
        )
        if additive_sum > 1.0 + _NORM_EPSILON:
            errors.append(
                f"Additive weights sum to {additive_sum:.4f}, "
                f"must be <= 1.0"
            )

        # Risk penalty must not exceed 1.0
        if self.w_risk > 1.0 + _NORM_EPSILON:
            errors.append(
                f"w_risk is {self.w_risk:.4f}, must be <= 1.0"
            )

        return errors


# ═══════════════════════════════════════════════════════════════════════════
# Result
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RankedCandidate:
    """Immutable result for a single ranked candidate.

    Attributes:
        candidate_id: Unique candidate identifier.
        final_score: Composite score in [0.0, 1.0] after clamping.
        norm_semantic_score: MinMax-normalized cross-encoder score [0, 1].
        availability_score: Stage 0B availability score [0, 1].
        evidence_coverage_score: Stage 0B evidence coverage score [0, 1].
        risk_score: Stage 0B risk score [0, 1].
        components: Breakdown of intermediate computation values.
        telemetry: Audit trail including defaults applied and warnings.
    """

    candidate_id: str
    final_score: float
    norm_semantic_score: float
    availability_score: float
    evidence_coverage_score: float
    risk_score: float
    components: Dict[str, Any]
    telemetry: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for downstream consumption."""
        return {
            "candidate_id": self.candidate_id,
            "final_score": round(self.final_score, 4),
            "norm_semantic_score": round(self.norm_semantic_score, 4),
            "availability_score": round(self.availability_score, 4),
            "evidence_coverage_score": round(
                self.evidence_coverage_score, 4
            ),
            "risk_score": round(self.risk_score, 4),
            "components": self.components,
            "telemetry": self.telemetry,
        }


# ═══════════════════════════════════════════════════════════════════════════
# FinalRanker
# ═══════════════════════════════════════════════════════════════════════════


class FinalRanker:
    """Stage 2 — Cross-Encoder re-ranking with behavioral score fusion.

    Takes the Top K candidates from Stage 1, computes deep semantic
    relevance via a Cross-Encoder, merges with Stage 0B behavioral
    intelligence profiles, and produces the final sorted ranking.

    Usage::

        ranker = FinalRanker()
        results = ranker.rank(jd_text, candidates, hireability_results)
        for r in results:
            print(r.candidate_id, r.final_score)

    Parameters
    ----------
    config : RankerConfig or None
        Configuration object. Uses defaults if None.
    """

    def __init__(self, config: Optional[RankerConfig] = None) -> None:
        """Initialize FinalRanker with configuration.

        Args:
            config: RankerConfig instance. Uses defaults if None.

        Raises:
            ValueError: If config fails validation.
        """
        self._config: RankerConfig = config or RankerConfig()
        errors: List[str] = self._config.validate()
        if errors:
            raise ValueError(
                f"Invalid RankerConfig: {'; '.join(errors)}"
            )

        # Lazy-loaded model — initialised on first call to rank()
        self._model: Optional[CrossEncoder] = None

    # ── Public API ───────────────────────────────────────────────────────

    def rank(
        self,
        jd_text: str,
        candidates: List[Dict[str, Any]],
        hireability_results: Dict[str, HireabilityResult],
    ) -> List[RankedCandidate]:
        """Rank candidates by fusing semantic relevance with behavioral scores.

        Parameters
        ----------
        jd_text : str
            The job description text to match against.
        candidates : list[dict]
            The subsetted list of raw candidate data from Stage 1.
            Each must contain at least ``candidate_id``.
        hireability_results : dict[str, HireabilityResult]
            Mapping of candidate_id to its Stage 0B HireabilityResult.
            Missing entries will receive pessimistic defaults.

        Returns
        -------
        list[RankedCandidate]
            Candidates sorted descending by final_score.
            Ties are broken by input order (stable sort).

        Raises
        ------
        ValueError
            If ``jd_text`` is empty or ``candidates`` is empty.
        """
        if not jd_text or not jd_text.strip():
            raise ValueError("jd_text must be a non-empty string")
        if not candidates:
            raise ValueError("candidates list must not be empty")

        n_candidates: int = len(candidates)
        logger.info(
            "Stage 2: Ranking %d candidates with cross-encoder",
            n_candidates,
        )

        # ── Step 1: Extract IDs and build search documents ───────────
        candidate_ids: List[str] = []
        search_documents: List[str] = []

        for candidate in candidates:
            cid: str = str(candidate.get("candidate_id", ""))
            candidate_ids.append(cid)
            search_documents.append(self._build_search_document(candidate))

        # ── Step 2: Cross-encoder scoring ────────────────────────────
        raw_scores: List[float] = self._compute_semantic_scores(
            jd_text, search_documents
        )

        # ── Step 3: MinMax normalization ─────────────────────────────
        norm_scores: List[float] = self._normalize_scores(raw_scores)

        # ── Step 4: Composite scoring and assembly ───────────────────
        ranked_candidates: List[RankedCandidate] = []

        # Track all defaults applied across the batch for top-level telemetry
        batch_defaults: List[Dict[str, Any]] = []

        for idx in range(n_candidates):
            cid = candidate_ids[idx]
            norm_semantic: float = norm_scores[idx]

            # Retrieve behavioral scores or apply pessimistic defaults
            telemetry: Dict[str, Any] = {
                "defaults_applied": [],
                "warnings": [],
                "ranking_timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            hr: Optional[HireabilityResult] = hireability_results.get(cid)

            if hr is not None:
                availability: float = hr.availability_score
                evidence: float = hr.evidence_coverage_score
                risk: float = hr.risk_score
            else:
                # Pessimistic defaults for missing behavioral data
                availability = 0.0
                evidence = 0.0
                risk = 1.0
                default_info: Dict[str, Any] = {
                    "candidate_id": cid,
                    "reason": "no_hireability_result",
                    "defaults_used": {
                        "availability_score": 0.0,
                        "evidence_coverage_score": 0.0,
                        "risk_score": 1.0,
                    },
                }
                telemetry["defaults_applied"].append(default_info)
                batch_defaults.append(default_info)
                telemetry["warnings"].append(
                    f"No HireabilityResult found for candidate "
                    f"'{cid}'; pessimistic defaults applied"
                )

            # Compute composite score
            raw_composite: float = self._compute_composite(
                norm_semantic, availability, evidence, risk
            )

            # Clamp to [0.0, 1.0]
            final_score: float = self._clamp(raw_composite, 0.0, 1.0)

            # Build component breakdown
            cfg = self._config
            components: Dict[str, Any] = {
                "raw_semantic_score": round(raw_scores[idx], 6),
                "norm_semantic_score": round(norm_semantic, 6),
                "weighted_semantic": round(
                    cfg.w_semantic * norm_semantic, 6
                ),
                "weighted_availability": round(
                    cfg.w_availability * availability, 6
                ),
                "weighted_evidence": round(
                    cfg.w_evidence * evidence, 6
                ),
                "weighted_risk_penalty": round(
                    cfg.w_risk * risk, 6
                ),
                "raw_composite": round(raw_composite, 6),
                "was_clamped": raw_composite != final_score,
            }

            ranked_candidates.append(
                RankedCandidate(
                    candidate_id=cid,
                    final_score=round(final_score, 4),
                    norm_semantic_score=round(norm_semantic, 4),
                    availability_score=round(availability, 4),
                    evidence_coverage_score=round(evidence, 4),
                    risk_score=round(risk, 4),
                    components=components,
                    telemetry=telemetry,
                )
            )

        # ── Step 5: Stable sort descending by final_score ────────────
        # Python's sorted() is guaranteed stable (Timsort), so equal
        # final_scores preserve the original input order from Stage 1.
        ranked_candidates.sort(
            key=lambda rc: rc.final_score, reverse=True
        )

        if batch_defaults:
            logger.warning(
                "Stage 2: %d candidates had no HireabilityResult; "
                "pessimistic defaults were applied",
                len(batch_defaults),
            )

        logger.info(
            "Stage 2: Ranking complete. Top candidate: %s (score=%.4f)",
            ranked_candidates[0].candidate_id if ranked_candidates else "N/A",
            ranked_candidates[0].final_score if ranked_candidates else 0.0,
        )

        return ranked_candidates

    # ── Private: Search Document Construction ────────────────────────────

    @staticmethod
    def _build_search_document(candidate: Dict[str, Any]) -> str:
        """Concatenate schema-verified fields into a single search string.

        Uses ONLY fields verified from the provided schema
        (identical to Stage 1 HybridRetriever._build_search_document):
            - profile.headline
            - profile.current_title
            - profile.summary
            - skills[*].name
            - career_history[*].title
            - career_history[*].company

        Missing or None fields are silently skipped.

        Parameters
        ----------
        candidate : dict
            A single candidate dict.

        Returns
        -------
        str
            Concatenated search document. May be empty if all fields
            are absent.
        """
        profile: Dict[str, Any] = candidate.get("profile") or {}

        # Extract profile text fields
        headline: str = str(profile.get("headline") or "")
        current_title: str = str(profile.get("current_title") or "")
        summary: str = str(profile.get("summary") or "")

        # Extract skill names
        skills: Sequence[Dict[str, Any]] = (
            candidate.get("skills") or []
        )
        skill_names: str = " ".join(
            str(skill.get("name") or "")
            for skill in skills
            if skill.get("name")
        )

        # Extract career history titles and companies
        career_history: Sequence[Dict[str, Any]] = (
            candidate.get("career_history") or []
        )
        career_parts: List[str] = []
        for entry in career_history:
            title: str = str(entry.get("title") or "")
            company: str = str(entry.get("company") or "")
            part: str = f"{title} {company}".strip()
            if part:
                career_parts.append(part)
        career_text: str = " ".join(career_parts)

        # Concatenate with field-order priority:
        # headline > current_title > summary > skills > career
        parts: List[str] = [
            p
            for p in [
                headline,
                current_title,
                summary,
                skill_names,
                career_text,
            ]
            if p
        ]
        return " ".join(parts)

    # ── Private: Semantic Scoring ────────────────────────────────────────

    def _load_model(self) -> CrossEncoder:
        """Load or return cached CrossEncoder model (CPU only).

        Returns
        -------
        CrossEncoder
            The loaded cross-encoder model on CPU device.
        """
        if self._model is None:
            logger.info(
                "Loading cross-encoder: %s (device=cpu)",
                self._config.model_name,
            )
            self._model = CrossEncoder(
                self._config.model_name, device="cpu"
            )
        return self._model

    def _compute_semantic_scores(
        self,
        jd_text: str,
        search_documents: List[str],
    ) -> List[float]:
        """Compute raw cross-encoder scores for JD × candidate pairs.

        Parameters
        ----------
        jd_text : str
            The job description text.
        search_documents : list[str]
            Search document for each candidate.

        Returns
        -------
        list[float]
            Raw logit scores from the cross-encoder. These are
            unbounded and must be normalized before fusion.
        """
        model: CrossEncoder = self._load_model()

        # Build sentence pairs: [(jd, doc1), (jd, doc2), ...]
        pairs: List[List[str]] = [
            [jd_text, doc] for doc in search_documents
        ]

        logger.info(
            "Cross-encoder predicting %d pairs", len(pairs)
        )

        # CrossEncoder.predict() returns numpy array of floats
        raw_predictions: np.ndarray = model.predict(pairs)

        # Ensure we always return a list of Python floats
        return [float(s) for s in raw_predictions]

    # ── Private: Score Normalization ─────────────────────────────────────

    @staticmethod
    def _normalize_scores(raw_scores: List[float]) -> List[float]:
        """MinMax-normalize raw cross-encoder logits to [0, 1].

        Edge cases:
            - If all scores are identical (max == min within epsilon),
              all candidates receive 1.0 to avoid artificial penalty.
            - Single candidate: receives 1.0 (degenerate case of above).

        Parameters
        ----------
        raw_scores : list[float]
            Raw logit scores from the cross-encoder.

        Returns
        -------
        list[float]
            Normalized scores in [0.0, 1.0].
        """
        if not raw_scores:
            return []

        min_score: float = min(raw_scores)
        max_score: float = max(raw_scores)
        score_range: float = max_score - min_score

        if score_range < _NORM_EPSILON:
            # Zero-variance: all candidates are indistinguishable.
            # Assign maximum semantic credit; let behavioral scores
            # break ties.
            return [1.0] * len(raw_scores)

        return [
            (score - min_score) / score_range for score in raw_scores
        ]

    # ── Private: Composite Scoring ───────────────────────────────────────

    def _compute_composite(
        self,
        norm_semantic: float,
        availability: float,
        evidence: float,
        risk: float,
    ) -> float:
        """Compute the raw composite score before clamping.

        Formula:
            Final = (w_semantic × norm_semantic)
                  + (w_availability × availability)
                  + (w_evidence × evidence)
                  - (w_risk × risk)

        Parameters
        ----------
        norm_semantic : float
            Normalized semantic score [0, 1].
        availability : float
            Stage 0B availability score [0, 1].
        evidence : float
            Stage 0B evidence coverage score [0, 1].
        risk : float
            Stage 0B risk score [0, 1].

        Returns
        -------
        float
            Raw composite score (may be outside [0, 1] before clamping).
        """
        cfg = self._config
        return (
            cfg.w_semantic * norm_semantic
            + cfg.w_availability * availability
            + cfg.w_evidence * evidence
            - cfg.w_risk * risk
        )

    # ── Private: Utility ─────────────────────────────────────────────────

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        """Clamp a numeric value to [lo, hi].

        Parameters
        ----------
        value : float
            Input value.
        lo : float
            Lower bound.
        hi : float
            Upper bound.

        Returns
        -------
        float
            Clamped value.
        """
        return max(lo, min(hi, float(value)))
