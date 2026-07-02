"""
Stage 0B — HireabilityEvaluator
================================
Team: AlgoRhythms | Author: Thejas J

Purpose:
    Process clean candidates (already passed Stage 0A IntegrityFilter)
    and calculate three continuous metrics (0.0 to 1.0):
        1. Availability_Score
        2. Evidence_Coverage_Score
        3. Risk_Score

    Stage 0B is strictly a behavioral and evidence-quality evaluator.
    Its outputs will later be consumed by ranking stages.

Prohibited dependencies:
    BM25, FAISS, Embeddings, Cross-Encoders,
    Semantic Similarity, Retrieval Scores, Ranking Scores.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════


@dataclass
class HireabilityConfig:
    """Configuration object for HireabilityEvaluator.

    Every weight, tier list, default value, and sentinel value resides
    here. No magic numbers should exist inside evaluation methods.
    """

    # ── Reference date for staleness calculations ────────────────
    reference_date: date = field(default_factory=lambda: date(2026, 6, 9))

    # ── Notice Period Multiplier ─────────────────────────────────
    notice_period_default: int = 90
    notice_period_tiers: List[Tuple[int, float]] = field(
        default_factory=lambda: [
            (30, 1.00),
            (60, 0.85),
            (90, 0.65),
            (120, 0.40),
        ]
    )
    notice_period_floor: float = 0.20

    # ── Last Active Multiplier ───────────────────────────────────
    last_active_default_days: int = 365
    last_active_tiers: List[Tuple[int, float]] = field(
        default_factory=lambda: [
            (30, 1.00),
            (90, 0.90),
            (180, 0.70),
            (365, 0.40),
        ]
    )
    last_active_floor: float = 0.15

    # ── Availability Score weights ───────────────────────────────
    availability_weight_notice: float = 0.50
    availability_weight_active: float = 0.50

    # ── Evidence Coverage Score ──────────────────────────────────
    evidence_verified_email_credit: float = 0.20
    evidence_github_credit: float = 0.25
    evidence_skills_credit: float = 0.30
    evidence_endorsement_credit: float = 0.15
    evidence_endorsement_threshold: int = 5
    evidence_profile_max_credit: float = 0.10
    evidence_profile_scale: float = 100.0

    # ── Risk Score ───────────────────────────────────────────────
    risk_weight_ghosting: float = 0.45
    risk_weight_flake: float = 0.35
    risk_weight_timeline: float = 0.20
    risk_default_response_rate: float = 0.0
    risk_default_completion_rate: float = 0.0

    # ── Sentinel values ──────────────────────────────────────────
    github_sentinel_value: float = -1.0

    def validate(self) -> List[str]:
        """Validate configuration integrity.

        Returns:
            List of human-readable error messages. Empty list = valid.
        """
        errors: List[str] = []

        # All weights must be non-negative
        weight_fields = [
            "availability_weight_notice",
            "availability_weight_active",
            "risk_weight_ghosting",
            "risk_weight_flake",
            "risk_weight_timeline",
            "evidence_verified_email_credit",
            "evidence_github_credit",
            "evidence_skills_credit",
            "evidence_endorsement_credit",
            "evidence_profile_max_credit",
        ]
        for attr_name in weight_fields:
            val = getattr(self, attr_name)
            if val < 0:
                errors.append(
                    f"{attr_name} must be non-negative, got {val}"
                )

        # Tier lists must be sorted ascending by threshold
        for tier_name in ("notice_period_tiers", "last_active_tiers"):
            tiers: List[Tuple[int, float]] = getattr(self, tier_name)
            thresholds = [t[0] for t in tiers]
            if thresholds != sorted(thresholds):
                errors.append(
                    f"{tier_name} must be sorted ascending by threshold"
                )

        # Evidence credits must sum to <= 1.0
        evidence_sum = (
            self.evidence_verified_email_credit
            + self.evidence_github_credit
            + self.evidence_skills_credit
            + self.evidence_endorsement_credit
            + self.evidence_profile_max_credit
        )
        if evidence_sum > 1.0 + 1e-9:
            errors.append(
                f"Evidence credits sum to {evidence_sum:.4f}, must be <= 1.0"
            )

        # Risk weights must sum to <= 1.0
        risk_sum = (
            self.risk_weight_ghosting
            + self.risk_weight_flake
            + self.risk_weight_timeline
        )
        if risk_sum > 1.0 + 1e-9:
            errors.append(
                f"Risk weights sum to {risk_sum:.4f}, must be <= 1.0"
            )

        return errors


# ═══════════════════════════════════════════════════════════════════
# Result
# ═══════════════════════════════════════════════════════════════════


@dataclass
class HireabilityResult:
    """Result of hireability evaluation for a single candidate.

    Attributes:
        candidate_id: Unique candidate identifier.
        availability_score: [0.0, 1.0] — higher = more available.
        evidence_coverage_score: [0.0, 1.0] — higher = more evidence exists.
        risk_score: [0.0, 1.0] — higher = more risky.
        components: Breakdown of intermediate values.
        telemetry: Warnings, defaults applied, audit data.
    """

    candidate_id: str
    availability_score: float
    evidence_coverage_score: float
    risk_score: float
    components: Dict[str, Any]
    telemetry: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for downstream consumption."""
        return {
            "candidate_id": self.candidate_id,
            "availability_score": round(self.availability_score, 4),
            "evidence_coverage_score": round(self.evidence_coverage_score, 4),
            "risk_score": round(self.risk_score, 4),
            "components": self.components,
            "telemetry": self.telemetry,
        }


# ═══════════════════════════════════════════════════════════════════
# Evaluator
# ═══════════════════════════════════════════════════════════════════


class HireabilityEvaluator:
    """Stage 0B behavioral and evidence-quality evaluator.

    Processes clean candidates (post Stage 0A) and computes:
        - Availability_Score   [0.0, 1.0]
        - Evidence_Coverage_Score [0.0, 1.0]
        - Risk_Score           [0.0, 1.0]

    No dependency on BM25, FAISS, embeddings, cross-encoders,
    semantic similarity, retrieval scores, or ranking scores.
    """

    def __init__(self, config: Optional[HireabilityConfig] = None) -> None:
        """Initialize evaluator with configuration.

        Args:
            config: HireabilityConfig instance. Uses defaults if None.

        Raises:
            ValueError: If config fails validation.
        """
        self._config = config or HireabilityConfig()
        errors = self._config.validate()
        if errors:
            raise ValueError(
                f"Invalid HireabilityConfig: {'; '.join(errors)}"
            )

    # ── Public API ───────────────────────────────────────────────

    def evaluate(self, candidate: Dict[str, Any]) -> HireabilityResult:
        """Evaluate a single candidate and return HireabilityResult.

        Args:
            candidate: Candidate dict conforming to Stage 0B schema.

        Returns:
            HireabilityResult with all three scores and telemetry.

        Raises:
            ValueError: If candidate_id is missing.
        """
        telemetry: Dict[str, Any] = {
            "defaults_applied": [],
            "warnings": [],
            "clamped_fields": [],
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
            "config_snapshot": {
                "reference_date": self._config.reference_date.isoformat(),
                "risk_weights": {
                    "ghosting": self._config.risk_weight_ghosting,
                    "flake": self._config.risk_weight_flake,
                    "timeline": self._config.risk_weight_timeline,
                },
                "availability_weights": {
                    "notice": self._config.availability_weight_notice,
                    "active": self._config.availability_weight_active,
                },
            },
            "input_hash": self._compute_input_hash(candidate),
        }

        # candidate_id is required — reject if absent
        candidate_id = candidate.get("candidate_id")
        if candidate_id is None:
            raise ValueError("candidate_id is required and must not be None")

        # Extract and default all signals
        signals = self._extract_signals(candidate, telemetry)

        # Compute Notice Period Multiplier (reused by Availability + Risk)
        npm = self._compute_notice_period_multiplier(
            signals["notice_period_days"]
        )

        # Compute Last Active Multiplier
        lam = self._compute_last_active_multiplier(
            signals["last_active_date"], telemetry
        )

        # Compute three output scores
        availability_score = self._compute_availability_score(npm, lam)
        evidence_coverage_score = self._compute_evidence_coverage_score(
            signals, telemetry
        )
        risk_score = self._compute_risk_score(signals, npm, telemetry)

        # Assemble component breakdown
        ghosting_risk = 1.0 - self._clamp(
            signals["recruiter_response_rate"], 0.0, 1.0
        )
        flake_risk = 1.0 - self._clamp(
            signals["interview_completion_rate"], 0.0, 1.0
        )
        timeline_risk = 1.0 - npm

        components: Dict[str, Any] = {
            "notice_period_multiplier": round(npm, 4),
            "last_active_multiplier": round(lam, 4),
            "ghosting_risk": round(ghosting_risk, 4),
            "flake_risk": round(flake_risk, 4),
            "timeline_risk": round(timeline_risk, 4),
            "evidence_breakdown": signals.get("_evidence_breakdown", {}),
        }

        return HireabilityResult(
            candidate_id=candidate_id,
            availability_score=round(availability_score, 4),
            evidence_coverage_score=round(evidence_coverage_score, 4),
            risk_score=round(risk_score, 4),
            components=components,
            telemetry=telemetry,
        )

    def evaluate_batch(
        self, candidates: List[Dict[str, Any]]
    ) -> List[HireabilityResult]:
        """Evaluate a batch of candidates.

        Args:
            candidates: List of candidate dicts.

        Returns:
            List of HireabilityResult in the same order.
        """
        return [self.evaluate(c) for c in candidates]

    # ── Private: Signal Extraction ───────────────────────────────

    def _extract_signals(
        self,
        candidate: Dict[str, Any],
        telemetry: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract and default all signals from candidate dict.

        Missing or None values are replaced with configured defaults
        and recorded in telemetry['defaults_applied'].
        """
        redrob = self._safe_get(candidate, "redrob_signals", None)
        if redrob is None:
            redrob = {}
            telemetry["defaults_applied"].append("redrob_signals")
        elif not isinstance(redrob, dict):
            telemetry["warnings"].append(
                "redrob_signals is not a dict, using empty dict"
            )
            redrob = {}

        signals: Dict[str, Any] = {}

        # notice_period_days
        npd = self._safe_get(redrob, "notice_period_days", None)
        if npd is None:
            npd = self._config.notice_period_default
            telemetry["defaults_applied"].append("notice_period_days")
        signals["notice_period_days"] = max(int(npd), 0)

        # last_active_date
        lad = self._safe_get(redrob, "last_active_date", None)
        if lad is None:
            telemetry["defaults_applied"].append("last_active_date")
        signals["last_active_date"] = lad

        # recruiter_response_rate
        rrr = self._safe_get(redrob, "recruiter_response_rate", None)
        if rrr is None:
            rrr = self._config.risk_default_response_rate
            telemetry["defaults_applied"].append("recruiter_response_rate")
        signals["recruiter_response_rate"] = rrr

        # interview_completion_rate
        icr = self._safe_get(redrob, "interview_completion_rate", None)
        if icr is None:
            icr = self._config.risk_default_completion_rate
            telemetry["defaults_applied"].append("interview_completion_rate")
        signals["interview_completion_rate"] = icr

        # skill_assessment_scores
        sas = self._safe_get(redrob, "skill_assessment_scores", None)
        if sas is None:
            sas = {}
            telemetry["defaults_applied"].append("skill_assessment_scores")
        elif not isinstance(sas, dict):
            telemetry["warnings"].append(
                f"skill_assessment_scores is not a dict "
                f"(got {type(sas).__name__}), treating as empty"
            )
            sas = {}
        signals["skill_assessment_scores"] = sas

        # endorsements_received
        er = self._safe_get(redrob, "endorsements_received", None)
        if er is None:
            er = 0
            telemetry["defaults_applied"].append("endorsements_received")
        signals["endorsements_received"] = max(int(er), 0)

        # profile_completeness_score
        pcs = self._safe_get(redrob, "profile_completeness_score", None)
        if pcs is None:
            pcs = 0.0
            telemetry["defaults_applied"].append("profile_completeness_score")
        signals["profile_completeness_score"] = float(pcs)

        # github_activity_score
        gas = self._safe_get(redrob, "github_activity_score", None)
        if gas is None:
            gas = self._config.github_sentinel_value
            telemetry["defaults_applied"].append("github_activity_score")
        signals["github_activity_score"] = float(gas)

        # verified_email (top-level, not inside redrob_signals)
        ve = self._safe_get(candidate, "verified_email", None)
        if ve is None:
            ve = False
            telemetry["defaults_applied"].append("verified_email")
        signals["verified_email"] = bool(ve)

        return signals

    # ── Private: Score Computation ───────────────────────────────

    def _compute_notice_period_multiplier(self, days: int) -> float:
        """Compute Notice Period Multiplier via tier lookup.

        Args:
            days: Notice period in days (already clamped to >= 0).

        Returns:
            Multiplier in [floor, 1.0].
        """
        return self._tier_lookup(
            days,
            self._config.notice_period_tiers,
            self._config.notice_period_floor,
        )

    def _compute_last_active_multiplier(
        self,
        date_str: Optional[str],
        telemetry: Dict[str, Any],
    ) -> float:
        """Compute Last Active Multiplier from date string.

        Args:
            date_str: ISO 8601 date string, or None if missing.
            telemetry: Telemetry dict for recording warnings.

        Returns:
            Multiplier in [floor, 1.0].
        """
        if date_str is None:
            inactive_days = self._config.last_active_default_days
        else:
            try:
                last_active = date.fromisoformat(str(date_str))
                delta = (self._config.reference_date - last_active).days
                if delta < 0:
                    telemetry["warnings"].append(
                        f"last_active_date '{date_str}' is in the future, "
                        f"treating as 0 inactive days"
                    )
                    delta = 0
                inactive_days = delta
            except (ValueError, TypeError) as exc:
                telemetry["warnings"].append(
                    f"last_active_date could not be parsed: '{date_str}' "
                    f"({exc}), using default "
                    f"{self._config.last_active_default_days} inactive days"
                )
                inactive_days = self._config.last_active_default_days

        return self._tier_lookup(
            inactive_days,
            self._config.last_active_tiers,
            self._config.last_active_floor,
        )

    def _compute_availability_score(
        self, npm: float, lam: float
    ) -> float:
        """Combine Notice Period Multiplier and Last Active Multiplier.

        Uses configurable weighted average.

        Args:
            npm: Notice Period Multiplier.
            lam: Last Active Multiplier.

        Returns:
            Availability Score in [0.0, 1.0].
        """
        w_np = self._config.availability_weight_notice
        w_la = self._config.availability_weight_active
        total_weight = w_np + w_la

        if total_weight == 0:
            return 0.0

        raw = (w_np * npm + w_la * lam) / total_weight
        return self._clamp(raw, 0.0, 1.0)

    def _compute_evidence_coverage_score(
        self,
        signals: Dict[str, Any],
        telemetry: Dict[str, Any],
    ) -> float:
        """Compute additive Evidence Coverage Score, capped at 1.0.

        Components:
            C1: verified_email == True         → +0.20
            C2: github_activity_score != -1    → +0.25
            C3: skill_assessment_scores != {}  → +0.30
            C4: endorsements_received > 5      → +0.15
            C5: profile_completeness normalized → +0.00 to +0.10

        Args:
            signals: Extracted signals dict.
            telemetry: Telemetry dict for recording clamping.

        Returns:
            Evidence Coverage Score in [0.0, 1.0].
        """
        cfg = self._config
        breakdown: Dict[str, float] = {}
        total = 0.0

        # C1: verified_email
        if signals["verified_email"] is True:
            credit = cfg.evidence_verified_email_credit
            breakdown["verified_email"] = credit
            total += credit
        else:
            breakdown["verified_email"] = 0.0

        # C2: github_activity_score != sentinel
        if signals["github_activity_score"] != cfg.github_sentinel_value:
            credit = cfg.evidence_github_credit
            breakdown["github_activity"] = credit
            total += credit
        else:
            breakdown["github_activity"] = 0.0

        # C3: skill_assessment_scores non-empty
        if signals["skill_assessment_scores"]:
            credit = cfg.evidence_skills_credit
            breakdown["skill_assessments"] = credit
            total += credit
        else:
            breakdown["skill_assessments"] = 0.0

        # C4: endorsements_received > threshold
        if signals["endorsements_received"] > cfg.evidence_endorsement_threshold:
            credit = cfg.evidence_endorsement_credit
            breakdown["endorsements"] = credit
            total += credit
        else:
            breakdown["endorsements"] = 0.0

        # C5: profile_completeness_score normalized
        raw_pcs = signals["profile_completeness_score"]
        if cfg.evidence_profile_scale > 0:
            normalized = self._clamp(
                raw_pcs / cfg.evidence_profile_scale, 0.0, 1.0
            )
        else:
            normalized = 0.0

        if raw_pcs < 0 or raw_pcs > cfg.evidence_profile_scale:
            clamped_val = self._clamp(
                raw_pcs, 0.0, cfg.evidence_profile_scale
            )
            telemetry["clamped_fields"].append(
                f"profile_completeness_score clamped from "
                f"{raw_pcs} to {clamped_val}"
            )

        credit = normalized * cfg.evidence_profile_max_credit
        breakdown["profile_completeness"] = round(credit, 4)
        total += credit

        # Store breakdown for component output
        signals["_evidence_breakdown"] = breakdown

        return self._clamp(total, 0.0, 1.0)

    def _compute_risk_score(
        self,
        signals: Dict[str, Any],
        npm: float,
        telemetry: Dict[str, Any],
    ) -> float:
        """Compute weighted Risk Score from component risks.

        Components:
            Ghosting_Risk  = 1 - recruiter_response_rate
            Flake_Risk     = 1 - interview_completion_rate
            Timeline_Risk  = 1 - Notice_Period_Multiplier

        Args:
            signals: Extracted signals dict.
            npm: Notice Period Multiplier (reused from availability).
            telemetry: Telemetry dict for recording clamping.

        Returns:
            Risk Score in [0.0, 1.0].
        """
        cfg = self._config

        # Ghosting Risk — defensive clamp on input rate
        rrr_raw = signals["recruiter_response_rate"]
        rrr = self._clamp(rrr_raw, 0.0, 1.0)
        if rrr_raw != rrr:
            telemetry["clamped_fields"].append(
                f"recruiter_response_rate clamped from {rrr_raw} to {rrr}"
            )
        ghosting_risk = 1.0 - rrr

        # Flake Risk — defensive clamp on input rate
        icr_raw = signals["interview_completion_rate"]
        icr = self._clamp(icr_raw, 0.0, 1.0)
        if icr_raw != icr:
            telemetry["clamped_fields"].append(
                f"interview_completion_rate clamped from {icr_raw} to {icr}"
            )
        flake_risk = 1.0 - icr

        # Timeline Risk — reuses Notice Period Multiplier
        timeline_risk = 1.0 - npm

        # Weighted combination
        raw = (
            cfg.risk_weight_ghosting * ghosting_risk
            + cfg.risk_weight_flake * flake_risk
            + cfg.risk_weight_timeline * timeline_risk
        )

        return self._clamp(raw, 0.0, 1.0)

    # ── Private: Utility Helpers ─────────────────────────────────

    @staticmethod
    def _tier_lookup(
        value: float,
        tiers: List[Tuple[int, float]],
        floor: float,
    ) -> float:
        """Look up a value against ascending tier thresholds.

        Evaluates tiers top-to-bottom. The first tier whose threshold
        is >= the value yields its multiplier. If no tier matches,
        the floor value is returned.

        Args:
            value: The value to look up.
            tiers: List of (threshold, multiplier) sorted ascending.
            floor: Value returned if no tier matches.

        Returns:
            The matched multiplier or floor.
        """
        for threshold, multiplier in tiers:
            if value <= threshold:
                return multiplier
        return floor

    @staticmethod
    def _safe_get(
        data: Dict[str, Any], key: str, default: Any
    ) -> Any:
        """Safely retrieve a key from a dict.

        Returns the default if the dict is not a dict, the key is
        absent, or the value is None.

        Args:
            data: Source dictionary.
            key: Key to retrieve.
            default: Fallback value.

        Returns:
            The value or default.
        """
        if not isinstance(data, dict):
            return default
        value = data.get(key)
        if value is None:
            return default
        return value

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        """Clamp a numeric value to [lo, hi].

        Args:
            value: Input value.
            lo: Lower bound.
            hi: Upper bound.

        Returns:
            Clamped value.
        """
        return max(lo, min(hi, float(value)))

    @staticmethod
    def _compute_input_hash(candidate: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of candidate input for traceability.

        Args:
            candidate: Raw candidate dict.

        Returns:
            Hex digest string, or 'hash_computation_failed' on error.
        """
        try:
            serialized = json.dumps(
                candidate, sort_keys=True, default=str
            )
            return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        except (TypeError, ValueError):
            return "hash_computation_failed"
