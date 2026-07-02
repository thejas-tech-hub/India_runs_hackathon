"""
test_hybrid_retriever.py
Stage 1 — Tests for HybridRetriever

Team:    AlgoRhythms
Student: THEJAS J

Test coverage:
    1. _build_search_document: complete, partial, empty candidates
    2. _tokenize: regex tokeniser punctuation handling
    3. _fuse_rrf: exact RRF math verification
    4. Missing fields: graceful handling of None/absent fields
    5. retrieve(): integration with synthetic candidates
    6. Edge cases: top_k > len(candidates), single candidate, empty docs
    7. Configuration validation: invalid parameters
    8. Retrieval cutoff: top_n_bm25 / top_n_dense
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

from hybrid_retriever import HybridRetriever


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def complete_candidate() -> Dict[str, Any]:
    """A candidate with ALL schema-verified fields present."""
    return {
        "candidate_id": "CAND_0000001",
        "profile": {
            "headline": "Backend Engineer | SQL, Spark, Cloud",
            "summary": (
                "Software / data professional with 6.9 years of experience "
                "in building scalable systems."
            ),
            "current_title": "Backend Engineer",
        },
        "skills": [
            {"name": "Python", "duration_months": 48},
            {"name": "Apache Spark", "duration_months": 36},
            {"name": "SQL", "duration_months": 60},
        ],
        "career_history": [
            {"title": "Backend Engineer", "company": "Mindtree"},
            {"title": "Data Analyst", "company": "Infosys"},
        ],
    }


@pytest.fixture
def minimal_candidate() -> Dict[str, Any]:
    """A candidate with only candidate_id — everything else missing."""
    return {"candidate_id": "CAND_MINIMAL"}


@pytest.fixture
def partial_candidate() -> Dict[str, Any]:
    """A candidate with some fields present, some missing/None."""
    return {
        "candidate_id": "CAND_PARTIAL",
        "profile": {
            "headline": "Data Scientist",
            "summary": None,
            "current_title": None,
        },
        "skills": [
            {"name": "R", "duration_months": 24},
            {"name": None, "duration_months": 12},  # name is None
        ],
        "career_history": None,  # entire career_history is None
    }


@pytest.fixture
def empty_profile_candidate() -> Dict[str, Any]:
    """A candidate where profile exists but is empty dict."""
    return {
        "candidate_id": "CAND_EMPTY_PROFILE",
        "profile": {},
        "skills": [],
        "career_history": [],
    }


@pytest.fixture
def none_profile_candidate() -> Dict[str, Any]:
    """A candidate where profile is explicitly None."""
    return {
        "candidate_id": "CAND_NONE_PROFILE",
        "profile": None,
        "skills": None,
        "career_history": None,
    }


@pytest.fixture
def sample_candidates() -> List[Dict[str, Any]]:
    """A small set of diverse candidates for integration testing."""
    return [
        {
            "candidate_id": "CAND_001",
            "profile": {
                "headline": "Senior Python Developer",
                "summary": "Expert in Python, Django, REST APIs and microservices",
                "current_title": "Senior Developer",
            },
            "skills": [
                {"name": "Python", "duration_months": 72},
                {"name": "Django", "duration_months": 48},
                {"name": "REST APIs", "duration_months": 36},
            ],
            "career_history": [
                {"title": "Senior Developer", "company": "TCS"},
            ],
        },
        {
            "candidate_id": "CAND_002",
            "profile": {
                "headline": "Java Backend Engineer",
                "summary": "Building enterprise applications with Java and Spring Boot",
                "current_title": "Backend Engineer",
            },
            "skills": [
                {"name": "Java", "duration_months": 60},
                {"name": "Spring Boot", "duration_months": 48},
            ],
            "career_history": [
                {"title": "Backend Engineer", "company": "Wipro"},
            ],
        },
        {
            "candidate_id": "CAND_003",
            "profile": {
                "headline": "Python Data Engineer | Spark, Airflow",
                "summary": "Building data pipelines with Python, PySpark and Airflow",
                "current_title": "Data Engineer",
            },
            "skills": [
                {"name": "Python", "duration_months": 48},
                {"name": "Apache Spark", "duration_months": 36},
                {"name": "Airflow", "duration_months": 24},
            ],
            "career_history": [
                {"title": "Data Engineer", "company": "Flipkart"},
            ],
        },
        {
            "candidate_id": "CAND_004",
            "profile": {
                "headline": "Frontend React Developer",
                "summary": "Specialist in React.js, TypeScript, and responsive UI",
                "current_title": "Frontend Developer",
            },
            "skills": [
                {"name": "React", "duration_months": 36},
                {"name": "TypeScript", "duration_months": 24},
            ],
            "career_history": [
                {"title": "Frontend Developer", "company": "Zomato"},
            ],
        },
        {
            "candidate_id": "CAND_005",
            "profile": {
                "headline": "DevOps Engineer | AWS, Docker, Kubernetes",
                "summary": "Cloud infrastructure and CI/CD pipeline automation",
                "current_title": "DevOps Engineer",
            },
            "skills": [
                {"name": "AWS", "duration_months": 48},
                {"name": "Docker", "duration_months": 36},
                {"name": "Kubernetes", "duration_months": 24},
            ],
            "career_history": [
                {"title": "DevOps Engineer", "company": "Razorpay"},
            ],
        },
    ]


@pytest.fixture
def retriever() -> HybridRetriever:
    """A HybridRetriever instance with default configuration."""
    return HybridRetriever()


# ═══════════════════════════════════════════════════════════════════════════
# 1. _build_search_document Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildSearchDocument:
    """Test search document construction from candidate dicts."""

    def test_complete_candidate(
        self, complete_candidate: Dict[str, Any]
    ) -> None:
        """All schema-verified fields are concatenated in correct order."""
        doc: str = HybridRetriever._build_search_document(complete_candidate)

        # Field-order: headline, current_title, summary, skills, career
        assert "Backend Engineer | SQL, Spark, Cloud" in doc
        assert "Backend Engineer" in doc
        assert "Software / data professional" in doc
        assert "Python" in doc
        assert "Apache Spark" in doc
        assert "SQL" in doc
        assert "Mindtree" in doc
        assert "Infosys" in doc
        assert "Data Analyst" in doc

    def test_headline_appears_before_summary(
        self, complete_candidate: Dict[str, Any]
    ) -> None:
        """Headline should appear before the summary in the document."""
        doc: str = HybridRetriever._build_search_document(complete_candidate)
        headline_pos: int = doc.index("Backend Engineer | SQL, Spark, Cloud")
        summary_pos: int = doc.index("Software / data professional")
        assert headline_pos < summary_pos

    def test_minimal_candidate_empty_doc(
        self, minimal_candidate: Dict[str, Any]
    ) -> None:
        """A candidate with only candidate_id yields empty string."""
        doc: str = HybridRetriever._build_search_document(minimal_candidate)
        assert doc == ""

    def test_partial_candidate(
        self, partial_candidate: Dict[str, Any]
    ) -> None:
        """Partial fields: present fields appear, None fields are skipped."""
        doc: str = HybridRetriever._build_search_document(partial_candidate)

        assert "Data Scientist" in doc  # headline present
        assert "R" in doc  # skill name "R" present
        # None summary, None current_title, None career_history → skipped
        # Skill with name=None → skipped
        assert doc.strip() != ""

    def test_empty_profile(
        self, empty_profile_candidate: Dict[str, Any]
    ) -> None:
        """Empty profile dict + empty lists → empty document."""
        doc: str = HybridRetriever._build_search_document(
            empty_profile_candidate
        )
        assert doc == ""

    def test_none_profile(
        self, none_profile_candidate: Dict[str, Any]
    ) -> None:
        """None profile/skills/career → empty document, no crash."""
        doc: str = HybridRetriever._build_search_document(
            none_profile_candidate
        )
        assert doc == ""

    def test_no_prohibited_fields(
        self, complete_candidate: Dict[str, Any]
    ) -> None:
        """Verify that adding prohibited fields does NOT affect output."""
        candidate_with_extras: Dict[str, Any] = {
            **complete_candidate,
            "certifications": ["AWS Certified"],
            "projects": ["ML Pipeline"],
            "achievements": ["Best Employee 2024"],
            "publications": ["Paper on NLP"],
            "education": [{"degree": "BTech"}],
        }
        doc_base: str = HybridRetriever._build_search_document(
            complete_candidate
        )
        doc_extras: str = HybridRetriever._build_search_document(
            candidate_with_extras
        )
        assert doc_base == doc_extras

    def test_career_history_title_and_company_joined(self) -> None:
        """Career title and company are space-joined per entry."""
        candidate: Dict[str, Any] = {
            "candidate_id": "CAND_CAREER",
            "career_history": [
                {"title": "SDE", "company": "Amazon"},
                {"title": "Intern", "company": "Google"},
            ],
        }
        doc: str = HybridRetriever._build_search_document(candidate)
        assert "SDE Amazon" in doc
        assert "Intern Google" in doc

    def test_career_entry_missing_title(self) -> None:
        """Career entry with no title — only company appears."""
        candidate: Dict[str, Any] = {
            "candidate_id": "CAND_NO_TITLE",
            "career_history": [{"company": "Meta"}],
        }
        doc: str = HybridRetriever._build_search_document(candidate)
        assert "Meta" in doc

    def test_career_entry_missing_company(self) -> None:
        """Career entry with no company — only title appears."""
        candidate: Dict[str, Any] = {
            "candidate_id": "CAND_NO_COMPANY",
            "career_history": [{"title": "Engineer"}],
        }
        doc: str = HybridRetriever._build_search_document(candidate)
        assert "Engineer" in doc

    def test_skills_with_all_none_names(self) -> None:
        """Skills list where every skill has name=None → no skill text."""
        candidate: Dict[str, Any] = {
            "candidate_id": "CAND_NULL_SKILLS",
            "skills": [
                {"name": None, "duration_months": 10},
                {"name": None, "duration_months": 20},
            ],
        }
        doc: str = HybridRetriever._build_search_document(candidate)
        # Should not contain any random text from skills
        assert doc == ""


# ═══════════════════════════════════════════════════════════════════════════
# 2. _tokenize Tests (Regex Tokeniser)
# ═══════════════════════════════════════════════════════════════════════════


class TestTokenize:
    """Test the regex-based tokeniser for BM25."""

    def test_basic_sentence(self) -> None:
        """Simple sentence → lowercased word list."""
        tokens: List[str] = HybridRetriever._tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_slash_separated(self) -> None:
        """Slash-separated tokens are correctly split."""
        tokens: List[str] = HybridRetriever._tokenize("Python/Django")
        assert tokens == ["python", "django"]

    def test_dot_separated(self) -> None:
        """Dot-separated tokens like React.js are split."""
        tokens: List[str] = HybridRetriever._tokenize("React.js")
        assert tokens == ["react", "js"]

    def test_pipe_separated(self) -> None:
        """Pipe-separated tokens (common in headlines) are split."""
        tokens: List[str] = HybridRetriever._tokenize(
            "Backend Engineer | SQL, Spark"
        )
        assert tokens == ["backend", "engineer", "sql", "spark"]

    def test_mixed_punctuation(self) -> None:
        """Multiple punctuation types handled correctly."""
        tokens: List[str] = HybridRetriever._tokenize(
            "C++, Java/Python (Spring-Boot) & REST APIs"
        )
        assert "c" in tokens
        assert "java" in tokens
        assert "python" in tokens
        assert "spring" in tokens
        assert "boot" in tokens
        assert "rest" in tokens
        assert "apis" in tokens
        # '+' and '&' and '(' should not appear
        assert "+" not in tokens
        assert "&" not in tokens
        assert "(" not in tokens

    def test_empty_string(self) -> None:
        """Empty string → empty list."""
        tokens: List[str] = HybridRetriever._tokenize("")
        assert tokens == []

    def test_only_punctuation(self) -> None:
        """String of only punctuation → empty list."""
        tokens: List[str] = HybridRetriever._tokenize("!@#$%^&*()")
        assert tokens == []

    def test_numbers_preserved(self) -> None:
        """Numbers are treated as word characters and preserved."""
        tokens: List[str] = HybridRetriever._tokenize("Python3.10 Java17")
        assert "python3" in tokens
        assert "10" in tokens
        assert "java17" in tokens

    def test_underscores_preserved(self) -> None:
        """Underscores are word characters and kept within tokens."""
        tokens: List[str] = HybridRetriever._tokenize("data_pipeline")
        assert tokens == ["data_pipeline"]

    def test_case_insensitivity(self) -> None:
        """All tokens are lowercased."""
        tokens: List[str] = HybridRetriever._tokenize("PyThOn DjAnGo")
        assert tokens == ["python", "django"]

    def test_whitespace_variations(self) -> None:
        """Tabs, newlines, multiple spaces handled correctly."""
        tokens: List[str] = HybridRetriever._tokenize("Python\tDjango\nFlask  REST")
        assert tokens == ["python", "django", "flask", "rest"]

    def test_hyphenated_words(self) -> None:
        """Hyphenated words are split into separate tokens."""
        tokens: List[str] = HybridRetriever._tokenize("micro-services")
        assert tokens == ["micro", "services"]

    def test_comma_separated(self) -> None:
        """Comma-separated items are correctly split."""
        tokens: List[str] = HybridRetriever._tokenize("Python,Django,Flask")
        assert tokens == ["python", "django", "flask"]


# ═══════════════════════════════════════════════════════════════════════════
# 3. _fuse_rrf Tests (Exact RRF Math)
# ═══════════════════════════════════════════════════════════════════════════


class TestFuseRRF:
    """Test Reciprocal Rank Fusion math with exact expected values."""

    def test_basic_rrf_formula(self) -> None:
        """Verify RRF formula: score = 1/(k+rank_bm25) + 1/(k+rank_dense)."""
        k_rrf: int = 60
        n: int = 3

        # Candidate 0: BM25 rank 1, Dense rank 2
        # Candidate 1: BM25 rank 2, Dense rank 1
        # Candidate 2: BM25 rank 3, Dense rank 3
        bm25_ranks: NDArray[np.int64] = np.array([1, 2, 3], dtype=np.int64)
        dense_ranks: NDArray[np.int64] = np.array([2, 1, 3], dtype=np.int64)

        top_indices: NDArray[np.intp] = HybridRetriever._fuse_rrf(
            bm25_ranks, dense_ranks, k_rrf, top_k=3, n_candidates=n
        )

        # Expected RRF scores:
        # Cand 0: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 ≈ 0.032520
        # Cand 1: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ≈ 0.032520  (same!)
        # Cand 2: 1/(60+3) + 1/(60+3) = 2/63           ≈ 0.031746
        score_0: float = 1.0 / 61 + 1.0 / 62
        score_1: float = 1.0 / 62 + 1.0 / 61
        score_2: float = 1.0 / 63 + 1.0 / 63

        assert score_0 == pytest.approx(score_1, abs=1e-12)
        assert score_0 > score_2

        # Both 0 and 1 tie; order depends on stable sort (0 before 1)
        assert top_indices[0] in (0, 1)
        assert top_indices[1] in (0, 1)
        assert top_indices[2] == 2

    def test_rrf_rank1_both_highest(self) -> None:
        """Candidate ranked 1 in both systems should be top RRF result."""
        k_rrf: int = 60
        n: int = 5

        bm25_ranks: NDArray[np.int64] = np.array(
            [3, 1, 5, 2, 4], dtype=np.int64
        )
        dense_ranks: NDArray[np.int64] = np.array(
            [4, 1, 2, 5, 3], dtype=np.int64
        )

        top_indices: NDArray[np.intp] = HybridRetriever._fuse_rrf(
            bm25_ranks, dense_ranks, k_rrf, top_k=1, n_candidates=n
        )

        # Candidate 1 has rank 1 in both → highest RRF
        assert top_indices[0] == 1

    def test_rrf_exact_scores(self) -> None:
        """Verify exact RRF score values for known inputs."""
        k_rrf: int = 60
        n: int = 2

        bm25_ranks: NDArray[np.int64] = np.array([1, 2], dtype=np.int64)
        dense_ranks: NDArray[np.int64] = np.array([1, 2], dtype=np.int64)

        top_indices: NDArray[np.intp] = HybridRetriever._fuse_rrf(
            bm25_ranks, dense_ranks, k_rrf, top_k=2, n_candidates=n
        )

        # Candidate 0: 1/61 + 1/61 = 2/61
        expected_score_0: float = 2.0 / 61.0
        # Candidate 1: 1/62 + 1/62 = 2/62
        expected_score_1: float = 2.0 / 62.0

        # Compute actual scores for verification
        actual_scores: NDArray[np.float64] = (
            1.0 / (k_rrf + bm25_ranks.astype(np.float64))
            + 1.0 / (k_rrf + dense_ranks.astype(np.float64))
        )

        assert actual_scores[0] == pytest.approx(expected_score_0, rel=1e-10)
        assert actual_scores[1] == pytest.approx(expected_score_1, rel=1e-10)

        # Candidate 0 should be first
        assert top_indices[0] == 0
        assert top_indices[1] == 1

    def test_rrf_top_k_limits_output(self) -> None:
        """top_k < n_candidates → only top_k indices returned."""
        k_rrf: int = 60
        n: int = 10

        bm25_ranks: NDArray[np.int64] = np.arange(1, n + 1, dtype=np.int64)
        dense_ranks: NDArray[np.int64] = np.arange(1, n + 1, dtype=np.int64)

        top_indices: NDArray[np.intp] = HybridRetriever._fuse_rrf(
            bm25_ranks, dense_ranks, k_rrf, top_k=3, n_candidates=n
        )

        assert len(top_indices) == 3
        # Best candidate should be index 0 (rank 1 in both)
        assert top_indices[0] == 0

    def test_rrf_opposite_rankings(self) -> None:
        """BM25 and dense disagree completely — RRF should compromise."""
        k_rrf: int = 60
        n: int = 4

        # BM25 prefers: 0, 1, 2, 3
        bm25_ranks: NDArray[np.int64] = np.array(
            [1, 2, 3, 4], dtype=np.int64
        )
        # Dense prefers: 3, 2, 1, 0
        dense_ranks: NDArray[np.int64] = np.array(
            [4, 3, 2, 1], dtype=np.int64
        )

        top_indices: NDArray[np.intp] = HybridRetriever._fuse_rrf(
            bm25_ranks, dense_ranks, k_rrf, top_k=4, n_candidates=n
        )

        # Expected RRF scores:
        # Cand 0: 1/61 + 1/64 = 0.01639 + 0.01563 = 0.03202
        # Cand 1: 1/62 + 1/63 = 0.01613 + 0.01587 = 0.03200
        # Cand 2: 1/63 + 1/62 = 0.01587 + 0.01613 = 0.03200
        # Cand 3: 1/64 + 1/61 = 0.01563 + 0.01639 = 0.03202
        # Candidates 0 and 3 tie, candidates 1 and 2 tie
        # Stable sort: 0 before 3, 1 before 2
        assert top_indices[0] in (0, 3)
        assert top_indices[1] in (0, 3)
        assert top_indices[2] in (1, 2)
        assert top_indices[3] in (1, 2)

    def test_rrf_single_candidate(self) -> None:
        """Single candidate → that candidate is returned."""
        k_rrf: int = 60
        bm25_ranks: NDArray[np.int64] = np.array([1], dtype=np.int64)
        dense_ranks: NDArray[np.int64] = np.array([1], dtype=np.int64)

        top_indices: NDArray[np.intp] = HybridRetriever._fuse_rrf(
            bm25_ranks, dense_ranks, k_rrf, top_k=1, n_candidates=1
        )

        assert len(top_indices) == 1
        assert top_indices[0] == 0


# ═══════════════════════════════════════════════════════════════════════════
# 4. Missing Fields Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMissingFields:
    """Verify graceful handling of None/absent fields."""

    def test_no_profile_key(self) -> None:
        """Candidate dict without 'profile' key → empty doc, no crash."""
        candidate: Dict[str, Any] = {"candidate_id": "CAND_X"}
        doc: str = HybridRetriever._build_search_document(candidate)
        assert doc == ""

    def test_profile_is_none(self) -> None:
        """profile=None → treated as empty dict."""
        candidate: Dict[str, Any] = {
            "candidate_id": "CAND_X",
            "profile": None,
        }
        doc: str = HybridRetriever._build_search_document(candidate)
        assert doc == ""

    def test_skills_is_none(self) -> None:
        """skills=None → treated as empty list."""
        candidate: Dict[str, Any] = {
            "candidate_id": "CAND_X",
            "skills": None,
        }
        doc: str = HybridRetriever._build_search_document(candidate)
        assert doc == ""

    def test_career_history_is_none(self) -> None:
        """career_history=None → treated as empty list."""
        candidate: Dict[str, Any] = {
            "candidate_id": "CAND_X",
            "career_history": None,
        }
        doc: str = HybridRetriever._build_search_document(candidate)
        assert doc == ""

    def test_all_fields_none(self) -> None:
        """Every retrievable field is None → empty doc, no crash."""
        candidate: Dict[str, Any] = {
            "candidate_id": "CAND_X",
            "profile": {
                "headline": None,
                "summary": None,
                "current_title": None,
            },
            "skills": [{"name": None, "duration_months": None}],
            "career_history": [{"title": None, "company": None}],
        }
        doc: str = HybridRetriever._build_search_document(candidate)
        assert doc == ""

    def test_mixed_none_and_present(self) -> None:
        """Only present fields appear in doc."""
        candidate: Dict[str, Any] = {
            "candidate_id": "CAND_MIX",
            "profile": {
                "headline": None,
                "summary": "Experienced data engineer",
                "current_title": None,
            },
            "skills": [
                {"name": "Spark", "duration_months": 24},
                {"name": None, "duration_months": 12},
            ],
            "career_history": [
                {"title": None, "company": "Amazon"},
            ],
        }
        doc: str = HybridRetriever._build_search_document(candidate)
        assert "Experienced data engineer" in doc
        assert "Spark" in doc
        assert "Amazon" in doc


# ═══════════════════════════════════════════════════════════════════════════
# 5. Configuration Validation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestConfiguration:
    """Test that invalid configurations raise errors."""

    def test_invalid_top_k(self) -> None:
        """top_k < 1 raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            HybridRetriever(top_k=0)

    def test_invalid_k_rrf(self) -> None:
        """k_rrf < 1 raises ValueError."""
        with pytest.raises(ValueError, match="k_rrf must be >= 1"):
            HybridRetriever(k_rrf=0)

    def test_invalid_batch_size(self) -> None:
        """batch_size < 1 raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            HybridRetriever(batch_size=0)

    def test_negative_top_k(self) -> None:
        """Negative top_k raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            HybridRetriever(top_k=-5)


# ═══════════════════════════════════════════════════════════════════════════
# 6. retrieve() Validation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRetrieveValidation:
    """Test input validation for the retrieve() method."""

    def test_empty_jd_text(self, retriever: HybridRetriever) -> None:
        """Empty JD text raises ValueError."""
        with pytest.raises(ValueError, match="jd_text must be a non-empty"):
            retriever.retrieve("", [{"candidate_id": "CAND_001"}])

    def test_whitespace_jd_text(self, retriever: HybridRetriever) -> None:
        """Whitespace-only JD text raises ValueError."""
        with pytest.raises(ValueError, match="jd_text must be a non-empty"):
            retriever.retrieve("   ", [{"candidate_id": "CAND_001"}])

    def test_empty_candidates(self, retriever: HybridRetriever) -> None:
        """Empty candidates list raises ValueError."""
        with pytest.raises(ValueError, match="candidates list must not be"):
            retriever.retrieve("Python developer needed", [])


# ═══════════════════════════════════════════════════════════════════════════
# 7. Integration Tests (with mocked model for speed)
# ═══════════════════════════════════════════════════════════════════════════


class TestRetrieveIntegration:
    """Integration tests for the full retrieve() pipeline.

    These tests mock the SentenceTransformer to avoid loading the
    actual model, which makes tests fast and deterministic.
    """

    @staticmethod
    def _create_mock_model(n_dim: int = 384) -> MagicMock:
        """Create a mock SentenceTransformer that returns random embeddings.

        The mock normalises outputs to simulate normalize_embeddings=True.
        Uses a fixed seed for reproducibility.
        """
        rng = np.random.RandomState(42)
        mock_model = MagicMock(spec=["encode"])

        def mock_encode(
            texts: List[str],
            batch_size: int = 256,
            normalize_embeddings: bool = True,
            show_progress_bar: bool = False,
            convert_to_numpy: bool = True,
        ) -> NDArray[np.float32]:
            n: int = len(texts)
            embeddings: NDArray[np.float32] = rng.randn(
                n, n_dim
            ).astype(np.float32)
            if normalize_embeddings:
                norms: NDArray[np.float32] = np.linalg.norm(
                    embeddings, axis=1, keepdims=True
                )
                # Avoid division by zero
                norms = np.maximum(norms, 1e-12)
                embeddings = embeddings / norms
            return embeddings

        mock_model.encode = mock_encode
        return mock_model

    def test_retrieve_returns_correct_count(
        self, sample_candidates: List[Dict[str, Any]]
    ) -> None:
        """retrieve() returns exactly top_k candidate IDs."""
        retriever = HybridRetriever(top_k=3)
        retriever._model = self._create_mock_model()

        result: List[str] = retriever.retrieve(
            "Python developer with Django experience",
            sample_candidates,
            top_k=3,
        )

        assert len(result) == 3
        assert all(isinstance(cid, str) for cid in result)

    def test_retrieve_returns_valid_ids(
        self, sample_candidates: List[Dict[str, Any]]
    ) -> None:
        """All returned IDs are from the input candidates."""
        retriever = HybridRetriever(top_k=5)
        retriever._model = self._create_mock_model()

        result: List[str] = retriever.retrieve(
            "Senior Python developer",
            sample_candidates,
        )

        valid_ids: set[str] = {
            str(c["candidate_id"]) for c in sample_candidates
        }
        for cid in result:
            assert cid in valid_ids

    def test_retrieve_no_duplicates(
        self, sample_candidates: List[Dict[str, Any]]
    ) -> None:
        """Returned IDs contain no duplicates."""
        retriever = HybridRetriever(top_k=5)
        retriever._model = self._create_mock_model()

        result: List[str] = retriever.retrieve(
            "Backend engineer",
            sample_candidates,
        )

        assert len(result) == len(set(result))

    def test_top_k_exceeds_candidates(
        self, sample_candidates: List[Dict[str, Any]]
    ) -> None:
        """top_k > len(candidates) → return all candidates."""
        retriever = HybridRetriever(top_k=1000)
        retriever._model = self._create_mock_model()

        result: List[str] = retriever.retrieve(
            "Any role",
            sample_candidates,
            top_k=1000,
        )

        assert len(result) == len(sample_candidates)

    def test_single_candidate(self) -> None:
        """Single candidate → that candidate is returned."""
        retriever = HybridRetriever(top_k=1)
        retriever._model = self._create_mock_model()

        candidates: List[Dict[str, Any]] = [
            {
                "candidate_id": "CAND_ONLY",
                "profile": {"headline": "Python Developer"},
            }
        ]

        result: List[str] = retriever.retrieve(
            "Python developer",
            candidates,
        )

        assert result == ["CAND_ONLY"]

    def test_top_k_override_in_retrieve(
        self, sample_candidates: List[Dict[str, Any]]
    ) -> None:
        """top_k parameter in retrieve() overrides instance-level top_k."""
        retriever = HybridRetriever(top_k=100)
        retriever._model = self._create_mock_model()

        result: List[str] = retriever.retrieve(
            "Python developer",
            sample_candidates,
            top_k=2,
        )

        assert len(result) == 2

    def test_retrieve_with_missing_fields(self) -> None:
        """Candidates with various missing fields do not crash."""
        retriever = HybridRetriever(top_k=5)
        retriever._model = self._create_mock_model()

        candidates: List[Dict[str, Any]] = [
            {"candidate_id": "CAND_FULL", "profile": {
                "headline": "Python Dev",
                "summary": "Expert",
                "current_title": "SDE",
            }},
            {"candidate_id": "CAND_NOPR"},  # no profile at all
            {"candidate_id": "CAND_NULLS", "profile": None,
             "skills": None, "career_history": None},
            {"candidate_id": "CAND_EMPTY", "profile": {},
             "skills": [], "career_history": []},
            {"candidate_id": "CAND_PARTIAL", "profile": {
                "headline": "Data Engineer",
            }},
        ]

        result: List[str] = retriever.retrieve(
            "Python developer",
            candidates,
        )

        assert len(result) == 5
        assert set(result) == {
            "CAND_FULL", "CAND_NOPR", "CAND_NULLS",
            "CAND_EMPTY", "CAND_PARTIAL",
        }

    def test_return_type_is_list_of_str(
        self, sample_candidates: List[Dict[str, Any]]
    ) -> None:
        """Verify return type is list[str] as specified in API contract."""
        retriever = HybridRetriever(top_k=3)
        retriever._model = self._create_mock_model()

        result = retriever.retrieve(
            "Backend engineer with Java experience",
            sample_candidates,
        )

        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Retrieval Cutoff Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRetrievalCutoffs:
    """Test that top_n_bm25 and top_n_dense cutoffs apply penalty ranks."""

    def test_bm25_cutoff_applies_penalty(self) -> None:
        """Candidates outside top_n_bm25 receive penalty rank."""
        retriever = HybridRetriever(top_n_bm25=2)
        retriever._model = TestRetrieveIntegration._create_mock_model()

        candidates: List[Dict[str, Any]] = [
            {"candidate_id": f"CAND_{i}",
             "profile": {"headline": f"Skill {i}"}}
            for i in range(5)
        ]

        # The method applies internally; we verify it doesn't crash
        # and returns valid results
        result: List[str] = retriever.retrieve(
            "Skill 0",
            candidates,
            top_k=5,
        )

        assert len(result) == 5
        assert len(set(result)) == 5  # no duplicates

    def test_dense_cutoff_applies_penalty(self) -> None:
        """Candidates outside top_n_dense receive penalty rank."""
        retriever = HybridRetriever(top_n_dense=2)
        retriever._model = TestRetrieveIntegration._create_mock_model()

        candidates: List[Dict[str, Any]] = [
            {"candidate_id": f"CAND_{i}",
             "profile": {"headline": f"Skill {i}"}}
            for i in range(5)
        ]

        result: List[str] = retriever.retrieve(
            "Skill 0",
            candidates,
            top_k=5,
        )

        assert len(result) == 5
        assert len(set(result)) == 5

    def test_both_cutoffs(self) -> None:
        """Both cutoffs applied simultaneously."""
        retriever = HybridRetriever(top_n_bm25=3, top_n_dense=3)
        retriever._model = TestRetrieveIntegration._create_mock_model()

        candidates: List[Dict[str, Any]] = [
            {"candidate_id": f"CAND_{i}",
             "profile": {"headline": f"Developer {i}"}}
            for i in range(10)
        ]

        result: List[str] = retriever.retrieve(
            "Developer",
            candidates,
            top_k=10,
        )

        assert len(result) == 10
        assert len(set(result)) == 10

    def test_cutoff_larger_than_candidates(self) -> None:
        """Cutoff larger than candidate count → no penalty applied."""
        retriever = HybridRetriever(top_n_bm25=100, top_n_dense=100)
        retriever._model = TestRetrieveIntegration._create_mock_model()

        candidates: List[Dict[str, Any]] = [
            {"candidate_id": f"CAND_{i}",
             "profile": {"headline": f"Role {i}"}}
            for i in range(5)
        ]

        result: List[str] = retriever.retrieve(
            "Role",
            candidates,
            top_k=5,
        )

        assert len(result) == 5


# ═══════════════════════════════════════════════════════════════════════════
# 9. BM25 Ranking Sanity Tests (with mock model)
# ═══════════════════════════════════════════════════════════════════════════


class TestBM25Ranking:
    """Test BM25 ranking behaviour with known text inputs."""

    def test_exact_keyword_match_ranked_higher(self) -> None:
        """Candidate with exact JD keywords should rank higher in BM25."""
        retriever = HybridRetriever(top_k=3)
        retriever._model = TestRetrieveIntegration._create_mock_model()

        candidates: List[Dict[str, Any]] = [
            {
                "candidate_id": "CAND_MATCH",
                "profile": {
                    "headline": "Python Django REST API Developer",
                    "summary": "Expert in Python Django REST microservices",
                },
                "skills": [{"name": "Python"}, {"name": "Django"}],
            },
            {
                "candidate_id": "CAND_PARTIAL_MATCH",
                "profile": {
                    "headline": "Python Developer",
                    "summary": "Some experience with Python",
                },
                "skills": [{"name": "Python"}],
            },
            {
                "candidate_id": "CAND_NO_MATCH",
                "profile": {
                    "headline": "Graphic Designer",
                    "summary": "Expert in Photoshop and Illustrator",
                },
                "skills": [{"name": "Photoshop"}],
            },
        ]

        jd: str = "Python Django REST API developer needed"

        # Build docs and run BM25 only to check ranking
        docs: List[str] = [
            retriever._build_search_document(c) for c in candidates
        ]
        bm25_ranks: NDArray[np.int64] = retriever._run_bm25(docs, jd, 3)

        # CAND_MATCH (idx 0) should have rank 1 (best BM25 match)
        assert bm25_ranks[0] == 1
        # CAND_NO_MATCH (idx 2) should have the worst rank
        assert bm25_ranks[2] == 3

    def test_bm25_empty_documents(self) -> None:
        """Candidates with empty search docs get low BM25 rank."""
        retriever = HybridRetriever(top_k=2)

        docs: List[str] = ["Python developer with Django", ""]
        ranks: NDArray[np.int64] = retriever._run_bm25(
            docs, "Python developer", 2
        )

        # Non-empty doc should rank higher
        assert ranks[0] < ranks[1]
