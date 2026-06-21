# Contract Risk Analysis Agent

Retrieval-grounded legal contract risk analysis system built using the CUAD (Contract Understanding Atticus Dataset), semantic embeddings, cosine similarity retrieval, explainability, and confidence scoring.

## Live Demo

Hugging Face Space:
https://harshithh-contract-risk-agent.hf.space

GitHub Repository:
https://github.com/hashith-reddy/contract-risk-agent

---

## Overview

This project analyzes contract clauses using semantic search and retrieval-based reasoning.

Instead of relying on keyword matching, the system converts contract clauses into vector embeddings and retrieves the most relevant legal clauses using cosine similarity. Risk assessments are generated using retrieved evidence, making every recommendation traceable and explainable.

---

## Features

* Contract clause chunking pipeline
* Semantic embeddings using all-MiniLM-L6-v2
* Cosine similarity retrieval engine
* Retrieval evaluation (Recall@1, Recall@3, Recall@5, MRR)
* Retrieval-grounded risk analysis
* Confidence scoring
* Explainability and evidence reporting
* Gradio web interface
* Hugging Face deployment

---

## System Architecture

Raw Contracts
↓
Contract Chunking
↓
Embedding Generation
↓
Vector Retrieval
↓
Risk Analysis Agent
↓
Explainability & Confidence Scoring
↓
Gradio Interface

---

## Project Results

Dataset Statistics:

* 1,387 contracts processed
* 1,743 semantic chunks generated
* 384-dimensional embeddings

Retrieval Metrics:

* Recall@1: 75.0%
* Recall@3: 87.5%
* Recall@5: 87.5%
* MRR: 81.25%

---

## Tech Stack

### AI / NLP

* Sentence Transformers
* all-MiniLM-L6-v2
* Semantic Retrieval
* Cosine Similarity Search

### Backend

* Python
* NumPy
* Pandas
* Scikit-Learn

### Data Processing

* Parquet
* LangChain Text Splitters

### UI & Deployment

* Gradio
* Hugging Face Spaces

---

## Repository Structure

contract-risk-agent/

├── src/data_prep/
├── src/embeddings/
├── src/retrieval/
├── src/risk_analysis/
├── data/
├── app.py
├── requirements.txt
└── README.md

---

## Example Analysis Output

Input:

"Neither party may assign this agreement without prior written consent."

Output:

* Risk Level: MEDIUM
* Confidence: 0.79
* Clause Type: Anti-Assignment
* Evidence: Retrieved similar clauses from CUAD dataset
* Recommendation: Review assignment restrictions and consent requirements

---

## Future Improvements

* Full online deployment with hosted embeddings
* Hybrid search (BM25 + embeddings)
* Cross-encoder reranking
* PDF contract upload support
* Multi-document contract comparison

---

## Author

Harshith Reddy

B.Tech CSE (AI)

ICFAI Foundation for Higher Education

Hyderabad, India
