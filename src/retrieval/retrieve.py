#!/usr/bin/env python3
"""
Retrieval engine that finds most relevant contract chunks using cosine similarity.
"""

import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_embeddings(embeddings_path):
    """Load embeddings from numpy file."""
    return np.load(embeddings_path)

def load_metadata(metadata_path):
    """Load metadata from parquet file."""
    return pd.read_parquet(metadata_path)

def load_chunks(chunks_path):
    """Load chunk text from parquet file."""
    return pd.read_parquet(chunks_path)

def embed_query(query, model):
    """Embed user query using the sentence transformer model."""
    return model.encode([query])

def find_relevant_chunks(embeddings, metadata, chunks, query_embedding, top_k=5):
    """Find top-k most relevant chunks using cosine similarity."""
    # Compute cosine similarities
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get indices of top-k similarities
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Get top-k chunks with their metadata and text
    results = []
    for idx in top_indices:
        chunk_metadata = metadata.iloc[idx]
        chunk_text = chunks[chunks['chunk_id'] == chunk_metadata['chunk_id']]['chunk_text'].values[0]
        result = {
            'similarity_score': similarities[idx],
            'contract_id': chunk_metadata['contract_id'],
            'clause_type': chunk_metadata['clause_type'],
            'chunk_text': chunk_text
        }
        results.append(result)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Retrieve most relevant contract chunks')
    parser.add_argument('--query', type=str, required=True, help='User query to search for')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top chunks to return')
    
    args = parser.parse_args()
    
    # Load embeddings and metadata
    embeddings_path = 'data/processed/cuad_embeddings.npy'
    metadata_path = 'data/processed/cuad_embedding_metadata.parquet'
    chunks_path = 'data/processed/cuad_chunks.parquet'
    
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    
    embeddings = load_embeddings(embeddings_path)
    metadata = load_metadata(metadata_path)
    chunks = load_chunks(chunks_path)
    
    # Load sentence transformer model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Embed the query
    query_embedding = embed_query(args.query, model)
    
    # Find relevant chunks
    results = find_relevant_chunks(embeddings, metadata, chunks, query_embedding, args.top_k)
    
    # Display results
    print(f"Top {args.top_k} most relevant contract chunks for query: '{args.query}'")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Similarity Score: {result['similarity_score']:.4f}")
        print(f"   Contract ID: {result['contract_id']}")
        print(f"   Clause Type: {result['clause_type']}")
        print(f"   Chunk Text: {result['chunk_text']}")
        print()

if __name__ == "__main__":
    main()
