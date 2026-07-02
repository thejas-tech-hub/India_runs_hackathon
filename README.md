# INDIA.RUNS Data & AI Challenge — Candidate Ranking Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![CPU Only](https://img.shields.io/badge/Compute-CPU%20Only-00897B?style=for-the-badge&logo=intel&logoColor=white)
![Runtime](https://img.shields.io/badge/Runtime-~254s-FF6F00?style=for-the-badge&logo=clockify&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-282%2F282%20Passed-4CAF50?style=for-the-badge&logo=pytest&logoColor=white)
![Submission](https://img.shields.io/badge/Submission-Ready-8E24AA?style=for-the-badge&logo=checkmarx&logoColor=white)

**Team:** AlgoRhythms  
**Challenge:** INDIA.RUNS (Redrob) Data & AI Hackathon

---

## Project Overview

An end-to-end, CPU-only candidate ranking pipeline that ingests **100,000 candidate profiles** and a Job Description, then produces a ranked shortlist of the **Top 100 candidates** with LLM-generated justifications for the Top 10.

The pipeline runs **entirely offline** — no external APIs, no GPU, no internet access at runtime.

---

## Features

- **Honeypot Detection** — Multi-signal integrity filter catches ~9.2% synthetic/adversarial candidates before they enter scoring
- **Hybrid Retrieval** — BM25 lexical search + dense embeddings (all-MiniLM-L6-v2) fused via Reciprocal Rank Fusion for robust recall
- **Cross-Encoder Re-ranking** — ms-marco-MiniLM-L-6-v2 provides deep semantic relevance scoring on the shortlist
- **Behavioral Intelligence** — Availability, Evidence Coverage, and Risk scores derived from career structure and profile completeness
- **Weighted Fusion** — Transparent composite formula combines semantic and behavioral signals into a single final score
- **LLM-Powered Justifications** — Phi-3-mini (Q4, CPU-only) generates structured, recruiter-ready explanations with GBNF grammar enforcement
- **Fully Offline** — Zero network calls, zero API keys, zero GPU — runs entirely on a standard CPU laptop
- **Reproducible** — Single command produces the identical `submission.csv` every time
- **Extensively Tested** — 282 unit tests covering all 5 pipeline stages

---

## Architecture

```
candidates.jsonl ─┐
                   ├──► Stage 0A: IntegrityFilter (Honeypot Detection)
                   │         │
                   │         ▼
                   ├──► Stage 0B: HireabilityEvaluator (Behavioral Scoring)
                   │         │
                   │         ▼
jd.txt ────────────┼──► Stage 1: HybridRetriever (BM25 + Dense + RRF → Top 500)
                   │         │
                   │         ▼
                   ├──► Stage 2: FinalRanker (Cross-Encoder + Fusion → Top 100)
                   │         │
                   │         ▼
model.gguf ────────┼──► Stage 3: ReasoningEngine (LLM Justifications → Top 10)
                   │         │
                   │         ▼
                   └──► Output: final_report.json + submission.csv
```

### Pipeline Stages

| Stage | Component | Description |
|-------|-----------|-------------|
| 0A | `IntegrityFilter` | Detects honeypot/synthetic candidates using skill-duration anomalies, career consistency checks, and field validation |
| 0B | `HireabilityEvaluator` | Computes Availability, Evidence Coverage, and Risk scores for each clean candidate |
| 1 | `HybridRetriever` | BM25 + Dense (all-MiniLM-L6-v2) + Reciprocal Rank Fusion to retrieve Top 500 |
| 2 | `FinalRanker` | Cross-Encoder (ms-marco-MiniLM-L-6-v2) re-ranking fused with behavioral scores |
| 3 | `ReasoningEngine` | Phi-3-mini-4k-instruct (Q4 GGUF) generates structured justifications for Top 10 |

---

## Folder Structure

```
india-runs-ai/
├── main.py                     # Master pipeline orchestrator
├── integrity_filter.py         # Stage 0A — Honeypot detection
├── hireability_evaluator.py    # Stage 0B — Behavioral scoring
├── hybrid_retriever.py         # Stage 1  — BM25 + Dense retrieval
├── final_ranker.py             # Stage 2  — Cross-encoder re-ranking
├── reasoning_engine.py         # Stage 3  — LLM justification
├── submission_exporter.py      # Official submission CSV exporter
├── requirements.txt            # Python dependencies
├── submission_metadata.yaml    # Competition metadata
├── README.md                   # This file
├── jd.txt                      # Job Description input
├── candidates.jsonl            # 100K candidate profiles (input data)
├── models/
│   └── Phi-3-mini-4k-instruct-q4.gguf  # Local LLM model (~2.4 GB)
├── final_report.json           # Pipeline output (detailed report)
├── submission.csv              # Official submission file
├── test_integrity_filter.py    # Unit tests — Stage 0A
├── test_hireability_evaluator.py # Unit tests — Stage 0B
├── test_hybrid_retriever.py    # Unit tests — Stage 1
├── test_final_ranker.py        # Unit tests — Stage 2
└── test_reasoning_engine.py    # Unit tests — Stage 3
```

---

## Installation

### Python Version

- **Python 3.10+** required

### Dependencies

```bash
pip install -r requirements.txt
```

### Required Input Files

| File | Description |
|------|-------------|
| `candidates.jsonl` | 100,000 candidate profiles in JSONL format |
| `jd.txt` | Job Description text file |
| `models/Phi-3-mini-4k-instruct-q4.gguf` | Quantized Phi-3 model for reasoning (~2.4 GB) |

---

## Reproducing the Submission

### Single Command

```bash
python main.py candidates.jsonl jd.txt models/Phi-3-mini-4k-instruct-q4.gguf
```

This will automatically produce:
- `final_report.json` — Full pipeline report with telemetry and justifications
- `submission.csv` — Official competition submission file (100 ranked candidates)

### Custom Output Paths

```bash
python main.py candidates.jsonl jd.txt models/Phi-3-mini-4k-instruct-q4.gguf \
    --output my_report.json \
    --submission-csv my_submission.csv
```

### Standalone CSV Export (from existing report)

```bash
python submission_exporter.py --input final_report.json --output submission.csv
```

---

## Runtime

| Metric | Value |
|--------|-------|
| **Total pipeline runtime** | ~254 seconds |
| **Compute** | CPU-only |
| **GPU required** | No |
| **External APIs** | None |
| **Internet required** | No |
| **Candidates processed** | 100,000 |

### Per-Stage Timing

| Stage | Duration |
|-------|----------|
| 0A — IntegrityFilter | ~4s |
| 0B — HireabilityEvaluator | ~13s |
| 1 — HybridRetriever | ~59s |
| 2 — FinalRanker | ~25s |
| 3 — ReasoningEngine | ~151s |

---

## Results

| Metric | Value |
|--------|-------|
| Total candidates ingested | 100,000 |
| Honeypots detected & filtered | 9,231 (9.2%) |
| Clean candidates evaluated | 90,769 |
| Top K retrieved (Stage 1) | 500 |
| Top N ranked (Stage 2) | 100 |
| Top N explained (Stage 3) | 10 |
| Total pipeline runtime | ~254 seconds |
| Unit tests | **282 / 282 passed** |

---

## Key Technical Details

- **Honeypot Detection**: Identifies ~9.2% synthetic candidates using multi-check integrity analysis
- **Hybrid Retrieval**: BM25 lexical + all-MiniLM-L6-v2 dense embeddings fused via Reciprocal Rank Fusion (k=60)
- **Cross-Encoder Re-ranking**: ms-marco-MiniLM-L-6-v2 (22M params) for deep semantic relevance
- **Fusion Formula**: `Final = 0.4×Semantic + 0.25×Availability + 0.2×Evidence - 0.15×Risk`
- **Reasoning Engine**: Phi-3-mini-4k-instruct (Q4_K_M quantization) with GBNF grammar-enforced JSON output
- **Memory Management**: Streaming ingestion + aggressive garbage collection to stay under 16 GB

---

## Running Tests

```bash
pytest test_integrity_filter.py test_hireability_evaluator.py \
       test_hybrid_retriever.py test_final_ranker.py \
       test_reasoning_engine.py -v
```

---

## Future Work

- **Multi-threaded reasoning** — Parallelize LLM justification generation across CPU cores to reduce Stage 3 latency
- **Confidence estimation** — Attach calibrated confidence intervals to final scores for downstream decision support
- **Multi-job ranking** — Extend the pipeline to rank candidates against multiple JDs simultaneously with shared embeddings
- **Incremental embedding updates** — Cache and incrementally update dense embeddings as new candidates arrive instead of full recomputation
- **Distributed execution** — Shard candidate processing across multiple machines for datasets beyond 100K

---

## Team

### AlgoRhythms

| Member | Role |
|--------|------|
| **THEJAS J** | Team Leader, AI/ML Engineer & System Architect |
| **Veekshith** | Backend & Infrastructure Engineer |
| **Raghavendra** | Data Engineering & Validation Engineer |
| **Abhijit** | Research & Quality Assurance Engineer |
