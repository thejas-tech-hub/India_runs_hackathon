"""
hybrid_retriever.py
Stage 1 — HybridRetriever for INDIA.RUNS Data & AI Challenge

Team:    AlgoRhythms
Student: THEJAS J

Purpose:
    Ingest a Job Description (JD) text and a list of clean candidates
    (already passed Stage 0A IntegrityFilter & Stage 0B HireabilityEvaluator).
    Return the candidate_ids of the Top K most relevant candidates
    using Hybrid Search (BM25 + Dense Retrieval + Reciprocal Rank Fusion).

Constraints:
    - CPU only (no GPU libraries)
    - No network access at runtime
    - Maximum 16 GB RAM
    - Target runtime < 60 seconds for 100K candidates
    - sentence-transformers/all-MiniLM-L6-v2 (384-dim, forced cpu)
    - faiss-cpu IndexFlatIP with L2-normalised vectors
    - rank_bm25 BM25Okapi with regex tokeniser
    - Reciprocal Rank Fusion with k_rrf = 60

Schema-verified fields used for search_document:
    profile.headline, profile.summary, profile.current_title,
    skills[*].name, career_history[*].title, career_history[*].company
"""
from __future__ import annotations

import logging
import re
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
from typing import Dict, List, Optional, Sequence

import faiss
import numpy as np
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_BATCH_SIZE: int = 256
_DEFAULT_TOP_K: int = 500
_DEFAULT_K_RRF: int = 60

# Regex tokeniser: splits on non-word boundaries, lowercased.
# Correctly handles "Python/Django" → ["python", "django"]
#                   "React.js"      → ["react", "js"]
_TOKEN_PATTERN: re.Pattern[str] = re.compile(r"\w+")


# ═══════════════════════════════════════════════════════════════════════════
# HybridRetriever
# ═══════════════════════════════════════════════════════════════════════════

class HybridRetriever:
    """Stage 1 — Hybrid BM25 + Dense retrieval with RRF fusion.

    Usage::

        retriever = HybridRetriever()
        top_ids = retriever.retrieve(jd_text, candidates, top_k=500)

    The retriever builds a BM25 index and a FAISS dense index over the
    candidates' search documents, scores the JD against both, and fuses
    the rankings via Reciprocal Rank Fusion.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for the sentence-transformer.
        Must be pre-downloaded to the local cache.
    batch_size : int
        Batch size for sentence-transformer encoding.
    top_k : int
        Default number of candidates to return.
    k_rrf : int
        RRF damping constant (default 60).
    top_n_bm25 : int or None
        If set, only the top N candidates from BM25 participate in RRF.
        Reduces fusion pool when N << total candidates.
    top_n_dense : int or None
        If set, only the top N candidates from dense retrieval
        participate in RRF.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        top_k: int = _DEFAULT_TOP_K,
        k_rrf: int = _DEFAULT_K_RRF,
        top_n_bm25: Optional[int] = None,
        top_n_dense: Optional[int] = 5000,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if k_rrf < 1:
            raise ValueError(f"k_rrf must be >= 1, got {k_rrf}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self._model_name: str = model_name
        self._batch_size: int = batch_size
        self._top_k: int = top_k
        self._k_rrf: int = k_rrf
        self._top_n_bm25: Optional[int] = top_n_bm25
        self._top_n_dense: Optional[int] = top_n_dense

        # Lazy-loaded model — initialised on first call to retrieve()
        self._model: Optional[SentenceTransformer] = None

    # ── Public API ───────────────────────────────────────────────────────

    def retrieve(
        self,
        jd_text: str,
        candidates: List[Dict[str, object]],
        top_k: Optional[int] = None,
    ) -> List[str]:
        """Return top-K candidate_ids ranked by hybrid RRF score.

        Parameters
        ----------
        jd_text : str
            The job description text to match against.
        candidates : list[dict]
            List of candidate dicts conforming to the Stage 1 schema.
            Each must contain at least ``candidate_id``.
        top_k : int or None
            Override the instance-level top_k for this call.

        Returns
        -------
        list[str]
            Ordered list of candidate_ids, most relevant first.
            Length is ``min(top_k, len(candidates))``.

        Raises
        ------
        ValueError
            If ``jd_text`` is empty or ``candidates`` is empty.
        """
        effective_top_k: int = top_k if top_k is not None else self._top_k

        if not jd_text or not jd_text.strip():
            raise ValueError("jd_text must be a non-empty string")
        if not candidates:
            raise ValueError("candidates list must not be empty")

        # Clamp top_k to candidate count
        n_candidates: int = len(candidates)
        effective_top_k = min(effective_top_k, n_candidates)

        # ── Step 1: Extract IDs and build search documents ───────────
        candidate_ids: List[str] = []
        search_documents: List[str] = []

        for candidate in candidates:
            cid: str = str(candidate.get("candidate_id", ""))
            candidate_ids.append(cid)
            search_documents.append(self._build_search_document(candidate))

        # ── Step 2: BM25 ranking ─────────────────────────────────────
        bm25_ranks: NDArray[np.int64] = self._run_bm25(
            search_documents, jd_text, n_candidates
        )

        # ── Step 3: Dense ranking ────────────────────────────────────
        dense_ranks: NDArray[np.int64] = self._run_dense(
            search_documents, jd_text, n_candidates
        )

        # ── Step 4: RRF fusion ───────────────────────────────────────
        top_indices: NDArray[np.intp] = self._fuse_rrf(
            bm25_ranks=bm25_ranks,
            dense_ranks=dense_ranks,
            k_rrf=self._k_rrf,
            top_k=effective_top_k,
            n_candidates=n_candidates,
        )

        return [candidate_ids[i] for i in top_indices]

    # ── Private: Search Document Construction ────────────────────────────

    @staticmethod
    def _build_search_document(candidate: Dict[str, object]) -> str:
        """Concatenate schema-verified fields into a single search string.

        Uses ONLY fields verified from the provided schema:
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
        profile: Dict[str, object] = candidate.get("profile") or {}  # type: ignore[assignment]

        # Extract profile text fields
        headline: str = str(profile.get("headline") or "")
        current_title: str = str(profile.get("current_title") or "")
        summary: str = str(profile.get("summary") or "")

        # Extract skill names
        skills: Sequence[Dict[str, object]] = candidate.get("skills") or []  # type: ignore[assignment]
        skill_names: str = " ".join(
            str(skill.get("name") or "")
            for skill in skills
            if skill.get("name")
        )

        # Extract career history titles and companies
        career_history: Sequence[Dict[str, object]] = (
            candidate.get("career_history") or []  # type: ignore[assignment]
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
            p for p in [headline, current_title, summary, skill_names, career_text]
            if p
        ]
        return " ".join(parts)

    # ── Private: Tokenisation ────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenise text using regex word extraction + lowercasing.

        Handles edge cases like:
            "Python/Django"  → ["python", "django"]
            "React.js"       → ["react", "js"]
            "C++"            → ["c"]   ('+' is non-word)
            ""               → []

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        list[str]
            List of lowercase tokens.
        """
        return re.findall(r"\w+", text.lower())

    # ── Private: BM25 Pipeline ───────────────────────────────────────────

    def _run_bm25(
        self,
        documents: List[str],
        query: str,
        n_candidates: int,
    ) -> NDArray[np.int64]:
        """Build BM25 index, score query, return 1-indexed rank array.

        Parameters
        ----------
        documents : list[str]
            Search documents for all candidates.
        query : str
            The JD text.
        n_candidates : int
            Total number of candidates.

        Returns
        -------
        NDArray[np.int64]
            Array of shape (n_candidates,) where ranks[i] is the
            1-indexed BM25 rank of candidate i.  If top_n_bm25 is set,
            candidates outside the top N receive rank = n_candidates + 1
            (penalty rank).
        """
        logger.info("Building BM25 index for %d candidates", n_candidates)

        tokenized_corpus: List[List[str]] = [
            self._tokenize(doc) for doc in documents
        ]
        tokenized_query: List[str] = self._tokenize(query)

        bm25: BM25Okapi = BM25Okapi(tokenized_corpus)
        scores: NDArray[np.float64] = bm25.get_scores(tokenized_query)

        # Argsort descending to get ranking
        sorted_indices: NDArray[np.intp] = np.argsort(-scores)

        # Build 1-indexed rank array for ALL candidates
        ranks: NDArray[np.int64] = np.empty(n_candidates, dtype=np.int64)
        ranks[sorted_indices] = np.arange(1, n_candidates + 1, dtype=np.int64)

        # Apply cutoff: candidates outside top_n_bm25 get penalty rank
        if self._top_n_bm25 is not None and self._top_n_bm25 < n_candidates:
            penalty_rank: int = n_candidates + 1
            penalty_indices: NDArray[np.intp] = sorted_indices[self._top_n_bm25:]
            ranks[penalty_indices] = penalty_rank

        logger.info("BM25 scoring complete")
        return ranks

    # ── Private: Dense Pipeline ──────────────────────────────────────────

    def _load_model(self) -> SentenceTransformer:
        """Load or return cached sentence-transformer model (CPU only).

        Returns
        -------
        SentenceTransformer
            The loaded model on CPU device.
        """
        if self._model is None:
            logger.info("Loading model: %s (device=cpu)", self._model_name)
            self._model = SentenceTransformer(
                self._model_name, device="cpu"
            )
        return self._model

    def _run_dense(
        self,
        documents: List[str],
        query: str,
        n_candidates: int,
    ) -> NDArray[np.int64]:
        """Encode documents, build/load FAISS index, score query, return ranks."""
        model: SentenceTransformer = self._load_model()

    # ── Candidate Embeddings Cache ──────────────────────────────
        embedding_file = "candidate_embeddings.npy"

        if os.path.exists(embedding_file):

            logger.info(
                "Loading cached embeddings from %s",
                embedding_file,
            )

            candidate_embeddings = np.load(
                embedding_file
            )

        else:

            logger.info(
                "Encoding %d candidates (batch_size=%d)",
                n_candidates,
                self._batch_size,
            )

            candidate_embeddings = model.encode(
                documents,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            np.save(
                embedding_file,
                candidate_embeddings,
            )

            logger.info(
                "Saved embeddings to %s",
                embedding_file,
            )

        candidate_embeddings = np.ascontiguousarray(
            candidate_embeddings,
            dtype=np.float32,
        )

        # ── Encode JD ───────────────────────────────────────────────
        jd_embedding: NDArray[np.float32] = model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        jd_embedding = np.ascontiguousarray(
            jd_embedding,
            dtype=np.float32,
        )

    # ── FAISS Cache ─────────────────────────────────────────────
        index_file = "candidate.index"

        if os.path.exists(index_file):

            logger.info(
                "Loading cached FAISS index"
            )

            index = faiss.read_index(index_file)

        else:

            d = candidate_embeddings.shape[1]

            index = faiss.IndexFlatIP(d)

            index.add(candidate_embeddings)

            faiss.write_index(
                index,
                index_file,
            )

            logger.info(
                "Saved FAISS index"
            )

    # ── Dense Retrieval ─────────────────────────────────────────
        retrieve_n: int = min(5000, n_candidates)

        scores: NDArray[np.float32]
        indices: NDArray[np.int64]

        scores, indices = index.search(
            jd_embedding,
            retrieve_n,
        )

        sorted_indices: NDArray[np.int64] = indices[0]

    # Everyone gets penalty rank by default
        ranks: NDArray[np.int64] = np.full(
            n_candidates,
            n_candidates + 1,
            dtype=np.int64,
        )

    # Retrieved candidates get actual ranks
        ranks[sorted_indices] = np.arange(
            1,
            len(sorted_indices) + 1,
            dtype=np.int64,
        )

        logger.info("Dense scoring complete")

        return ranks

    # ── Private: Reciprocal Rank Fusion ──────────────────────────────────

    @staticmethod
    def _fuse_rrf(
        bm25_ranks: NDArray[np.int64],
        dense_ranks: NDArray[np.int64],
        k_rrf: int,
        top_k: int,
        n_candidates: int,
    ) -> NDArray[np.intp]:
        """Compute RRF scores and return top-K candidate indices.

        RRF_Score(c) = 1/(k_rrf + rank_bm25(c)) + 1/(k_rrf + rank_dense(c))

        Parameters
        ----------
        bm25_ranks : NDArray[np.int64]
            1-indexed BM25 ranks, shape (n_candidates,).
        dense_ranks : NDArray[np.int64]
            1-indexed dense ranks, shape (n_candidates,).
        k_rrf : int
            RRF damping constant.
        top_k : int
            Number of candidates to return.
        n_candidates : int
            Total number of candidates.

        Returns
        -------
        NDArray[np.intp]
            Indices of top-K candidates sorted by descending RRF score.
        """
        rrf_scores: NDArray[np.float64] = (
            1.0 / (k_rrf + bm25_ranks.astype(np.float64))
            + 1.0 / (k_rrf + dense_ranks.astype(np.float64))
        )

        # Argsort descending, take top_k
        top_indices: NDArray[np.intp] = np.argsort(-rrf_scores)[:top_k]
        return top_indices
