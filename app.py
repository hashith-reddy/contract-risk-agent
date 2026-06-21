#!/usr/bin/env python3
"""
Gradio interface for the Contract Risk Analysis Agent.
This file creates a simple UI that allows users to paste contract text and get risk analysis.
"""

import gradio as gr
import sys
import os
import json

# Add the project root to the path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the contract analysis functionality
from src.risk_analysis.analyze_contract import (
    load_embeddings, 
    load_metadata, 
    load_chunks,
    embed_query,
    find_relevant_chunks,
    calculate_confidence_score,
    determine_risk_level,
    generate_explanation,
    get_retrieval_statistics,
    get_confidence_level,
    generate_risk_report
)
from sentence_transformers import SentenceTransformer

# Global model for embedding - loaded once for efficiency
model = None

def analyze_contract(contract_text, top_k=5):
    """
    Analyze contract text and return structured risk analysis results.
    
    Args:
        contract_text (str): The contract text to analyze
        top_k (int): Number of top chunks to retrieve
        
    Returns:
        dict: Structured risk analysis results
    """
    global model
    
    # Check if we have the required data files - they are in the root directory 
    embeddings_path = 'data/processed/cuad_embeddings.npy'
    metadata_path = 'data/processed/cuad_embedding_metadata.parquet'
    chunks_path = 'data/processed/cuad_chunks.parquet'
    
    # Validate that required files exist
    if not os.path.exists(embeddings_path):
        return {
            "risk_level": "Error",
            "confidence": "N/A",
            "confidence_level": "N/A",
            "clause_types": [],
            "retrieval_stats": {},
            "evidence": [],
            "recommendations": [f"Error: Embeddings file not found at {embeddings_path}"]
        }
    if not os.path.exists(metadata_path):
        return {
            "risk_level": "Error",
            "confidence": "N/A",
            "confidence_level": "N/A",
            "clause_types": [],
            "retrieval_stats": {},
            "evidence": [],
            "recommendations": [f"Error: Metadata file not found at {metadata_path}"]
        }
    if not os.path.exists(chunks_path):
        return {
            "risk_level": "Error",
            "confidence": "N/A",
            "confidence_level": "N/A",
            "clause_types": [],
            "retrieval_stats": {},
            "evidence": [],
            "recommendations": [f"Error: Chunks file not found at {chunks_path}"]
        }
    
    try:
        # Load data
        embeddings = load_embeddings(embeddings_path)
        metadata = load_metadata(metadata_path)
        chunks = load_chunks(chunks_path)
        
        # Initialize model if not already done
        if model is None:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Embed the query
        query_embedding = embed_query(contract_text, model)
        
        # Find relevant chunks
        results = find_relevant_chunks(embeddings, metadata, chunks, query_embedding, top_k)
        
        # Generate structured risk report components
        risk_level = determine_risk_level(results)
        confidence_score = calculate_confidence_score(results)
        confidence_level = get_confidence_level(confidence_score)
        retrieval_stats = get_retrieval_statistics(results)
        clause_types = list(set([r['clause_type'] for r in results]))
        evidence = results
        recommendations = generate_explanation(contract_text, results)
        
        return {
            "risk_level": risk_level,
            "confidence": f"{confidence_score:.2f}",
            "confidence_level": confidence_level,
            "clause_types": clause_types,
            "retrieval_stats": retrieval_stats,
            "evidence": evidence,
            "recommendations": recommendations
        }
        
    except Exception as e:
        return {
            "risk_level": "Error",
            "confidence": "N/A",
            "confidence_level": "N/A",
            "clause_types": [],
            "retrieval_stats": {},
            "evidence": [],
            "recommendations": [f"Error during analysis: {str(e)}"]
        }

def format_output(results):
    """Format the results for display in Gradio components."""
    if results["risk_level"] == "Error":
        return f"Error occurred during analysis:\n\n{results['recommendations'][0]}"
    
    output = []
    output.append(f"## Risk Level: {results['risk_level']}")
    output.append(f"## Confidence Score: {results['confidence']}")
    output.append(f"## Confidence Level: {results['confidence_level']}")
    output.append(f"## Detected Clause Types: {', '.join(results['clause_types']) if results['clause_types'] else 'None'}")
    
    output.append("\n### Retrieval Statistics")
    for key, value in results['retrieval_stats'].items():
        output.append(f"- {key}: {value}")
    
    output.append("\n### Evidence")
    if not results['evidence']:
        output.append("No relevant evidence found.")
    else:
        for i, evidence_item in enumerate(results['evidence'], 1):
            output.append(f"**Evidence {i}:**")
            output.append(f"- Similarity Score: {evidence_item.get('similarity_score', 'N/A'):.2f}")
            output.append(f"- Contract ID: {evidence_item.get('contract_id', 'N/A')}")
            output.append(f"- Clause Type: {evidence_item.get('clause_type', 'N/A')}")
            output.append(f"- Chunk Text: {evidence_item.get('chunk_text', 'N/A')}")
            output.append("")
    
    output.append("\n### Recommendations")
    if not results['recommendations']:
        output.append("No specific recommendations generated.")
    else:
        for i, rec in enumerate(results['recommendations'], 1):
            output.append(f"{i}. {rec}")
    
    return "\n".join(output)

def format_structured_output(results):
    """Format the results for structured display components."""
    if results["risk_level"] == "Error":
        return (
            results["risk_level"],
            results["confidence"], 
            results["confidence_level"],
            ", ".join(results['clause_types']) if results['clause_types'] else "None",
            json.dumps(results['retrieval_stats'], indent=2),
            "\n\n".join([f"**Evidence {i}:**\n- Similarity Score: {e.get('similarity_score', 'N/A')}\n- Contract ID: {e.get('contract_id', 'N/A')}\n- Clause Type: {e.get('clause_type', 'N/A')}\n- Chunk Text: {e.get('chunk_text', 'N/A')}" for i, e in enumerate(results['evidence'], 1)]),
            "\n\n".join(results['recommendations'])
        )
    
    return (
        results["risk_level"],
        results["confidence"], 
        results["confidence_level"],
        ", ".join(results['clause_types']) if results['clause_types'] else "None",
        json.dumps(results['retrieval_stats'], indent=2),
        "\n\n".join([f"**Evidence {i}:**\n- Similarity Score: {e.get('similarity_score', 'N/A')}\n- Contract ID: {e.get('contract_id', 'N/A')}\n- Clause Type: {e.get('clause_type', 'N/A')}\n- Chunk Text: {e.get('chunk_text', 'N/A')}" for i, e in enumerate(results['evidence'], 1)]),
        "\n\n".join(results['recommendations'])
    )

# Create Gradio interface
with gr.Blocks(title="Contract Risk Analysis Agent") as demo:
    gr.Markdown("# Contract Risk Analysis Agent")
    gr.Markdown("Retrieval-grounded legal contract risk analysis using CUAD, semantic embeddings, and evidence-based reasoning.")
    
    with gr.Row():
        with gr.Column(scale=1):
            contract_input = gr.Textbox(
                label="Contract Text", 
                placeholder="Paste your contract text here...",
                lines=15
            )
            top_k_slider = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Top K Chunks to Retrieve")
            analyze_button = gr.Button("Analyze Contract")
            
            # Example buttons
            gr.Markdown("## Examples")
            with gr.Row():
                example_btn1 = gr.Button("Example 1")
                example_btn2 = gr.Button("Example 2")
                example_btn3 = gr.Button("Example 3")
            
            # Example textboxes (hidden but used for loading)
            example1_textbox = gr.Textbox(value="Neither party may assign this agreement without prior written consent.", visible=False)
            example2_textbox = gr.Textbox(value="The agreement automatically renews for successive one year terms unless terminated.", visible=False)  
            example3_textbox = gr.Textbox(value="Liability shall not exceed fees paid under this agreement.", visible=False)
            
        with gr.Column(scale=2):
            # Structured output components
            risk_level_output = gr.Textbox(label="Risk Level", interactive=False)
            confidence_output = gr.Textbox(label="Confidence Score", interactive=False)
            confidence_level_output = gr.Textbox(label="Confidence Level", interactive=False)
            clause_types_output = gr.Textbox(label="Detected Clause Types", interactive=False)
            retrieval_stats_output = gr.Textbox(label="Retrieval Statistics", interactive=False)
            evidence_output = gr.Textbox(label="Evidence", lines=10, interactive=False)
            recommendations_output = gr.Textbox(label="Recommendations", lines=10, interactive=False)
    
    # Example button event handlers
    example_btn1.click(fn=lambda: example1_textbox.value, inputs=[], outputs=contract_input)
    example_btn2.click(fn=lambda: example2_textbox.value, inputs=[], outputs=contract_input)
    example_btn3.click(fn=lambda: example3_textbox.value, inputs=[], outputs=contract_input)
    
    # Analyze button event handler
    analyze_button.click(
        fn=analyze_contract,
        inputs=[contract_input, top_k_slider],
        outputs=None  # Will use the structured components directly
    ).then(
        fn=format_structured_output,
        inputs=None,
        outputs=[
            risk_level_output,
            confidence_output,
            confidence_level_output,
            clause_types_output,
            retrieval_stats_output,
            evidence_output,
            recommendations_output
        ]
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch()
