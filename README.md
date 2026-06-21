# Contract Risk Analysis Agent

This project implements a contract risk analysis system that uses semantic search to identify relevant contract clauses based on user queries. The system processes contracts, generates embeddings, and enables efficient retrieval of relevant contract information.

## System Architecture

The system consists of three main components:

1. **Data Preparation** (`src/data_prep/chunk_contracts.py`)
   - Splits contract clauses into manageable text chunks
   - Uses RecursiveCharacterTextSplitter for intelligent chunking
   - Generates deterministic chunk IDs for consistency

2. **Embeddings Generation** (`src/embeddings/generate_embeddings.py`)
   - Uses sentence-transformers to create semantic embeddings
   - Employs the 'all-MiniLM-L6-v2' model for efficient processing
   - Stores embeddings and metadata in parquet files

3. **Retrieval Engine** (`src/retrieval/retrieve.py`)
   - Implements cosine similarity search for relevant chunks
   - Supports command-line queries with configurable results count
   - Returns contract ID, clause type, and chunk text with similarity scores

## How to Use

### Prerequisites
Install required dependencies:
```bash
pip install -r requirements.txt
```

### Data Preparation
1. Place your raw contract data in `data/raw/` directory as a parquet file named `cuad_labels.parquet`
2. Run the chunking process:
```bash
python src/data_prep/chunk_contracts.py
```

### Embeddings Generation
3. Generate embeddings for the chunks:
```bash
python src/embeddings/generate_embeddings.py
```

### Retrieval
4. Query the system:
```bash
python src/retrieval/retrieve.py --query "What are the termination conditions?"
```

## Features

- Semantic search using sentence transformers
- Configurable chunk size and overlap for text splitting
- Efficient cosine similarity computation
- Detailed result information including contract ID, clause type, and similarity score
- Deterministic chunk IDs for reproducible results
- Command-line interface for easy querying
- Gradio UI for interactive analysis
- Structured output with risk level, confidence, evidence, and recommendations

## File Structure

```
contract-risk-agent/
├── src/
│   ├── data_prep/
│   │   └── chunk_contracts.py
│   ├── embeddings/
│   │   └── generate_embeddings.py
│   └── retrieval/
│       └── retrieve.py
├── data/
│   ├── raw/              # Raw contract data
│   └── processed/        # Processed chunks, embeddings, and metadata
├── app.py                # Gradio UI application
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## Dependencies

The project requires the following Python packages (listed in `requirements.txt`):
- gradio
- sentence-transformers
- numpy
- pandas
- pyarrow
- scikit-learn
- langchain-text-splitters

## Model Information

The system uses the 'sentence-transformers/all-MiniLM-L6-v2' model for embedding generation, which provides a good balance between performance and accuracy for semantic similarity tasks.

## Output Format

When querying, the system returns:
- Risk Level (Low, Medium, High)
- Confidence Score (0-1)
- Confidence Level (Low, Medium, High)
- Detected Clause Types
- Retrieval Statistics 
- Evidence with similarity scores and chunk details
- Recommendations based on detected clauses

## Deployment for Hugging Face Spaces

This project is ready to be deployed on Hugging Face Spaces. For deployment instructions, see `README_HF.md`.