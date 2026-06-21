import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import argparse

# Load the embeddings and metadata
def load_data():
    print("Loading data...")
    
    # Load chunks with their metadata
    chunks_df = pd.read_parquet('data/processed/cuad_chunks.parquet')
    
    # Load embeddings
    embeddings = np.load('data/processed/cuad_embeddings.npy')
    
    # Load embedding metadata
    embedding_metadata = pd.read_parquet('data/processed/cuad_embedding_metadata.parquet')
    
    print(f"Loaded {len(chunks_df)} chunks")
    print(f"Loaded {len(embeddings)} embeddings")
    print(f"Loaded {len(embedding_metadata)} embedding metadata records")
    
    return chunks_df, embeddings, embedding_metadata

# Create a simple retrieval function using cosine similarity
def retrieve_chunks(query, embeddings, chunk_embeddings, top_k=5):
    # For simplicity, we'll use the sentence transformer model to encode the query
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode the query
    query_embedding = model.encode([query])
    
    # Calculate cosine similarities
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    return top_indices, similarities[top_indices]

# Calculate recall metrics
def calculate_recall(retrieved_clause_types, relevant_clause_type):
    """
    Calculate Recall@1, Recall@3, Recall@5
    """
    # Check if the relevant clause type is in the retrieved results
    recall_at_1 = 1.0 if relevant_clause_type in retrieved_clause_types[:1] else 0.0
    recall_at_3 = 1.0 if relevant_clause_type in retrieved_clause_types[:3] else 0.0
    recall_at_5 = 1.0 if relevant_clause_type in retrieved_clause_types[:5] else 0.0
    
    return recall_at_1, recall_at_3, recall_at_5

# Calculate MRR (Mean Reciprocal Rank)
def calculate_mrr(retrieved_clause_types, relevant_clause_type):
    """
    Calculate Mean Reciprocal Rank
    """
    try:
        # Find the rank of the relevant clause type (1-indexed)
        rank = retrieved_clause_types.index(relevant_clause_type) + 1
        mrr = 1.0 / rank
    except ValueError:
        # If not found, MRR is 0
        mrr = 0.0
    
    return mrr

# Main evaluation function
def evaluate_retrieval():
    print("Starting retrieval evaluation...")
    
    # Load data
    chunks_df, embeddings, embedding_metadata = load_data()
    
    # Get unique clause types
    clause_types = chunks_df['clause_type'].unique()
    print(f"Clause types: {list(clause_types)}")
    
    # Initialize results storage
    all_results = []
    per_clause_type_results = {}
    
    # Create a model for encoding queries
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # For each clause type, use it as a query and evaluate retrieval
    for clause_type in clause_types:
        print(f"\nEvaluating for clause type: {clause_type}")
        
        # Get chunks that belong to this clause type (ground truth)
        relevant_chunks = chunks_df[chunks_df['clause_type'] == clause_type]
        
        if len(relevant_chunks) == 0:
            print(f"No chunks found for clause type {clause_type}")
            continue
            
        # Create a query using the clause type name
        query = clause_type
        
        # Get top-k retrieved chunks (using all embeddings)
        top_indices, similarities = retrieve_chunks(query, embeddings, embeddings, top_k=5)
        
        # Get the clause types of retrieved chunks
        retrieved_clause_types = []
        for idx in top_indices:
            if idx < len(chunks_df):
                retrieved_clause_types.append(chunks_df.iloc[idx]['clause_type'])
        
        print(f"Query: {query}")
        print(f"Relevant chunks count: {len(relevant_chunks)}")
        print(f"Retrieved clause types: {retrieved_clause_types}")
        
        # Calculate metrics
        recall_at_1, recall_at_3, recall_at_5 = calculate_recall(retrieved_clause_types, clause_type)
        mrr = calculate_mrr(retrieved_clause_types, clause_type)
        
        # Store results
        per_clause_type_results[clause_type] = {
            "recall_at_1": recall_at_1,
            "recall_at_3": recall_at_3,
            "recall_at_5": recall_at_5,
            "mrr": mrr
        }
        
        all_results.append({
            "query": query,
            "relevant_clause_type": clause_type,
            "retrieved_clause_types": retrieved_clause_types,
            "recall_at_1": recall_at_1,
            "recall_at_3": recall_at_3,
            "recall_at_5": recall_at_5,
            "mrr": mrr
        })
    
    # Calculate overall metrics
    overall_recall_at_1 = np.mean([r["recall_at_1"] for r in all_results])
    overall_recall_at_3 = np.mean([r["recall_at_3"] for r in all_results])
    overall_recall_at_5 = np.mean([r["recall_at_5"] for r in all_results])
    overall_mrr = np.mean([r["mrr"] for r in all_results])
    
    # Create final results structure
    results = {
        "overall": {
            "recall_at_1": float(overall_recall_at_1),
            "recall_at_3": float(overall_recall_at_3),
            "recall_at_5": float(overall_recall_at_5),
            "mrr": float(overall_mrr)
        },
        "per_clause_type": per_clause_type_results
    }
    
    # Save results to file
    os.makedirs('eval/results', exist_ok=True)
    with open('eval/results/retrieval_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Overall Recall@1: {overall_recall_at_1:.4f}")
    print(f"Overall Recall@3: {overall_recall_at_3:.4f}")
    print(f"Overall Recall@5: {overall_recall_at_5:.4f}")
    print(f"Overall MRR: {overall_mrr:.4f}")
    
    print("\nPer-clause-type metrics:")
    print("-" * 60)
    print(f"{'Clause Type':<25} {'Recall@1':<10} {'Recall@3':<10} {'Recall@5':<10} {'MRR':<10}")
    print("-" * 60)
    
    for clause_type, metrics in per_clause_type_results.items():
        print(f"{clause_type:<25} {metrics['recall_at_1']:<10.4f} {metrics['recall_at_3']:<10.4f} {metrics['recall_at_5']:<10.4f} {metrics['mrr']:<10.4f}")
    
    return results

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate retrieval quality')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top chunks to retrieve (default: 5)')
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_retrieval()
    
    print("\nEvaluation complete!")