#!/usr/bin/env python3
"""
Contract Risk Analysis Agent that uses retrieval results as evidence.
"""

import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import List, Dict, Any
import sys

# Add the project root to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_embeddings(embeddings_path: str) -> np.ndarray:
    """Load embeddings from numpy file."""
    return np.load(embeddings_path)

def load_metadata(metadata_path: str) -> pd.DataFrame:
    """Load metadata from parquet file."""
    return pd.read_parquet(metadata_path)

def load_chunks(chunks_path: str) -> pd.DataFrame:
    """Load chunk text from parquet file."""
    return pd.read_parquet(chunks_path)

def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """Embed user query using the sentence transformer model."""
    return model.encode([query])

def find_relevant_chunks(embeddings: np.ndarray, metadata: pd.DataFrame, chunks: pd.DataFrame, 
                        query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
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

def determine_risk_level(evidence: List[Dict]) -> str:
    """
    Determine risk level based on retrieved evidence.
    
    Args:
        evidence: List of retrieved clauses with their similarity scores and types
        
    Returns:
        Risk level: LOW, MEDIUM, or HIGH
    """
    if not evidence:
        return "LOW"
    
    # Calculate risk based on average similarity score and clause characteristics
    # This approach is purely retrieval-grounded without hardcoded mappings
    
    # Get all similarity scores
    similarity_scores = [item['similarity_score'] for item in evidence]
    
    # Calculate average similarity
    avg_similarity = np.mean(similarity_scores)
    
    # Determine risk level based on similarity thresholds (more semantic similarity = higher potential risk)
    if avg_similarity >= 0.75:
        # High similarity suggests strong relevance to known risky clauses
        return "HIGH"
    elif avg_similarity >= 0.60:
        # Moderate similarity suggests moderate risk
        return "MEDIUM"
    else:
        # Low similarity suggests lower risk
        return "LOW"

def generate_risk_report(query: str, evidence: List[Dict]) -> str:
    """
    Generate a structured risk report based on retrieved evidence.
    
    Args:
        query: The contract text being analyzed
        evidence: List of retrieved clauses with their similarity scores and types
        
    Returns:
        Formatted risk report as string
    """
    risk_level = determine_risk_level(evidence)
    
    # Get unique clause types
    clause_types = list(set(item['clause_type'] for item in evidence))
    
    # Build the report
    report = "==================================================\n"
    report += "CONTRACT RISK ANALYSIS\n"
    report += "======================\n\n"
    
    report += f"Risk Level:\n{risk_level}\n\n"
    
    report += "Detected Clause Types:\n\n"
    for clause_type in sorted(clause_types):
        report += f"* {clause_type}\n"
    report += "\n"
    
    report += "Evidence\n\n"
    
    for i, item in enumerate(evidence, 1):
        report += f"Clause Type: {item['clause_type']}\n"
        report += f"Similarity Score: {item['similarity_score']:.4f}\n"
        report += f"Evidence Snippet: {item['chunk_text']}\n\n"
    
    # Generate risk summary
    report += "Risk Summary\n\n"
    if not evidence:
        report += "No relevant clauses found in the database. No significant risks identified.\n\n"
    else:
        report += "The retrieved clauses indicate potential risk factors based on their type and content:\n\n"
        
        # Group by clause type for better summary
        clause_groups = {}
        for item in evidence:
            if item['clause_type'] not in clause_groups:
                clause_groups[item['clause_type']] = []
            clause_groups[item['clause_type']].append(item)
            
        for clause_type, items in clause_groups.items():
            report += f"- {clause_type}: "
            # Check for specific risk indicators
            if 'Anti-Assignment' in clause_type:
                report += "Restricts assignment of contract rights, which may limit flexibility.\n"
            elif 'Cap On Liability' in clause_type:
                report += "Limits liability exposure, which could be favorable or unfavorable depending on context.\n"
            elif 'Renewal Term' in clause_type:
                report += "Specifies automatic renewal terms, potentially creating ongoing obligations.\n"
            elif 'Termination' in clause_type:
                report += "Contains termination conditions that may create risks for either party.\n"
            elif 'Governing Law' in clause_type:
                report += "Specifies governing law, which could impact dispute resolution and legal risk.\n"
            else:
                report += "Relevant clause type detected.\n"
    
    # Generate recommendations
    report += "\nRecommendations\n\n"
    if not evidence:
        report += "No specific recommendations as no relevant clauses were found in the database.\n"
    else:
        report += "Based on the retrieved evidence, consider the following:\n\n"
        
        for clause_type, items in clause_groups.items():
            # Add specific recommendation based on clause type
            if 'Anti-Assignment' in clause_type:
                report += f"• Review assignment restrictions in {clause_type} clauses. Ensure proper consent procedures are documented.\n"
            elif 'Cap On Liability' in clause_type:
                report += f"• Evaluate the liability cap in {clause_type} clauses. Consider whether it provides adequate protection.\n"
            elif 'Renewal Term' in clause_type:
                report += f"• Assess automatic renewal terms in {clause_type} clauses. Determine if there are adequate termination rights.\n"
            elif 'Termination' in clause_type:
                report += f"• Examine termination conditions in {clause_type} clauses. Ensure they provide sufficient protection for your interests.\n"
            elif 'Governing Law' in clause_type:
                report += f"• Consider the implications of governing law in {clause_type} clauses. Verify it aligns with your jurisdictional strategy.\n"
            else:
                report += f"• Review {clause_type} clauses carefully as they may contain relevant terms for legal review.\n"
    
    return report

def main():
    parser = argparse.ArgumentParser(description='Analyze contract text for risk using retrieval-based evidence')
    parser.add_argument('--text', type=str, required=True, help='Contract text to analyze')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top chunks to retrieve')
    
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
    query_embedding = embed_query(args.text, model)
    
    # Find relevant chunks
    results = find_relevant_chunks(embeddings, metadata, chunks, query_embedding, args.top_k)
    
    # Generate and print risk report
    report = generate_risk_report(args.text, results)
    print(report)

if __name__ == "__main__":
    main()