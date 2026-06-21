# ⚖️ Contract Risk Analysis Agent

A retrieval-grounded legal AI system that analyzes contract clauses using semantic search, vector embeddings, explainable risk assessment, and evidence-based reasoning.

Built using the CUAD (Contract Understanding Atticus Dataset), Sentence Transformers, and Gradio, this project demonstrates an end-to-end AI pipeline from contract preprocessing to deployment.

---

## 🚀 Live Demo

**Hugging Face Deployment**

https://harshithh-contract-risk-agent.hf.space

**GitHub Repository**

https://github.com/hashith-reddy/contract-risk-agent

---

## 📌 Project Overview

Contract review is a time-consuming process that requires identifying critical legal clauses such as:

* Anti-Assignment
* Change of Control
* Cap on Liability
* Uncapped Liability
* Renewal Terms
* Non-Compete
* Termination for Convenience
* Most Favored Nation

This system uses semantic retrieval instead of simple keyword matching to identify relevant clauses and generate explainable risk assessments supported by retrieved evidence.

Every recommendation produced by the system is traceable to actual contract clauses retrieved from the CUAD dataset.

---

## 🏗 System Architecture

```text
Raw Contract Data
        │
        ▼
Contract Processing
        │
        ▼
Text Chunking
        │
        ▼
Embedding Generation
        │
        ▼
Vector Retrieval Engine
        │
        ▼
Risk Analysis Agent
        │
        ▼
Explainability Layer
        │
        ▼
Gradio User Interface
```

---

## ✨ Key Features

### Semantic Contract Search

* Sentence Transformer embeddings
* Cosine similarity retrieval
* Top-K clause ranking
* Retrieval-grounded reasoning

### Risk Analysis

* Risk classification
* Confidence scoring
* Evidence-backed recommendations
* Clause type detection

### Explainability

* Similarity score reporting
* Evidence ranking
* Retrieval statistics
* Classification rationale

### Evaluation

* Recall@1
* Recall@3
* Recall@5
* Mean Reciprocal Rank (MRR)

### Deployment

* Gradio web application
* Hugging Face Spaces deployment
* Public demo access

---

## 📊 Dataset & Results

### Dataset

**CUAD (Contract Understanding Atticus Dataset)**

Processing Results:

| Metric              | Value            |
| ------------------- | ---------------- |
| Contracts Processed | 1,387            |
| Chunks Generated    | 1,743            |
| Embedding Dimension | 384              |
| Embedding Model     | all-MiniLM-L6-v2 |

### Retrieval Performance

| Metric   | Score  |
| -------- | ------ |
| Recall@1 | 75.0%  |
| Recall@3 | 87.5%  |
| Recall@5 | 87.5%  |
| MRR      | 81.25% |

---

## 🧠 Technology Stack

### AI / Machine Learning

* Sentence Transformers
* all-MiniLM-L6-v2
* Semantic Retrieval
* Vector Search
* Cosine Similarity

### Data Engineering

* Pandas
* NumPy
* PyArrow
* Parquet

### NLP Processing

* LangChain Text Splitters
* Recursive Character Chunking

### User Interface

* Gradio

### Deployment

* Hugging Face Spaces
* GitHub

---

## 📂 Repository Structure

```text
contract-risk-agent/
│
├── src/
│   ├── data_prep/
│   │   └── chunk_contracts.py
│   │
│   ├── embeddings/
│   │   └── generate_embeddings.py
│   │
│   ├── retrieval/
│   │   └── retrieve.py
│   │
│   └── risk_analysis/
│       └── analyze_contract.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── eval/
│   └── evaluate_retrieval.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 🔍 Example Analysis

### Input

```text
Neither party may assign this agreement without prior written consent.
```

### Output

```text
Risk Level: MEDIUM
Confidence: 0.79
Clause Type: Anti-Assignment

Evidence:
5 of top 5 retrieved clauses matched Anti-Assignment clauses.

Recommendation:
Review assignment restrictions and ensure appropriate consent procedures are documented.
```

---

## 📈 Development Milestones

* Phase 1 – CUAD Data Processing
* Phase 2 – Contract Chunking Pipeline
* Phase 3 – Embedding Generation
* Phase 4 – Semantic Retrieval Engine
* Phase 5 – Retrieval Evaluation
* Phase 6 – Risk Analysis Agent
* Phase 7 – Explainability & Confidence Scoring
* Phase 8 – Gradio User Interface
* Phase 9 – Deployment & Hugging Face Integration

---

## 🔮 Future Improvements

* Hybrid Retrieval (BM25 + Embeddings)
* Cross-Encoder Reranking
* PDF Contract Upload Support
* Multi-Contract Comparison
* Clause Highlighting
* Legal Knowledge Graph Integration

---

## 👨‍💻 Author

**Harshith Reddy**

B.Tech Computer Science & Engineering (AI)

ICFAI Foundation for Higher Education (IFHE)

Hyderabad, India

---

## 📜 License

MIT License
