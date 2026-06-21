"""
Chunk contracts from the CUAD dataset.
"""
import argparse
import hashlib
import os
import pandas as pd
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_contracts(input_file, output_file, chunk_size=1000, chunk_overlap=200):
    """
    Chunk contracts from the CUAD dataset.
    
    Args:
        input_file (str): Path to input parquet file
        output_file (str): Path to output parquet file
        chunk_size (int): Maximum chunk size in characters
        chunk_overlap (int): Overlap between chunks in characters
    """
    # Read the input data
    df = pd.read_parquet(input_file)
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Process each row and create chunks
    chunked_rows = []
    
    for idx, row in df.iterrows():
        # Get the clause text
        clause_text = row['clause_text']
        
        if pd.isna(clause_text) or not str(clause_text).strip():
            continue
            
        # Split into chunks
        chunks = text_splitter.split_text(str(clause_text))
        
        # Create a deterministic chunk_id for each chunk
        contract_id = row['contract_id']
        clause_type = row['clause_type']
        
        for i, chunk in enumerate(chunks):
            # Create deterministic chunk_id using hash of contract_id + clause_type + chunk content
            chunk_content = f"{contract_id}_{clause_type}_{chunk}"
            chunk_id = hashlib.md5(chunk_content.encode()).hexdigest()
            
            chunked_rows.append({
                'chunk_id': chunk_id,
                'contract_id': contract_id,
                'clause_type': clause_type,
                'chunk_text': chunk,
                'annotation': row['annotation'],
                'is_risky': row['is_risky']
            })
    
    # Create DataFrame with chunked data
    chunked_df = pd.DataFrame(chunked_rows)
    
    # Save to parquet
    chunked_df.to_parquet(output_file, index=False)
    
    print(f"Processed {len(chunked_df)} chunks and saved to {output_file}")
    return chunked_df

def main():
    """Main function to run the chunking process."""
    parser = argparse.ArgumentParser(description="Chunk contracts from CUAD dataset")
    parser.add_argument("--input", default="data/processed/cuad_labels.parquet", help="Input parquet file")
    parser.add_argument("--output", default="data/processed/cuad_chunks.parquet", help="Output parquet file")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap in characters")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Process the contracts
    chunked_df = chunk_contracts(
        args.input, 
        args.output, 
        args.chunk_size, 
        args.chunk_overlap
    )
    
    # Show verification information
    print("\nVerification:")
    print(f"Shape: {chunked_df.shape}")
    print(f"Head:")
    print(chunked_df.head())
    
    # Calculate chunk statistics
    chunk_lengths = chunked_df['chunk_text'].str.len()
    print(f"\nChunk length statistics:")
    print(f"Average: {chunk_lengths.mean():.2f}")
    print(f"Min: {chunk_lengths.min()}")
    print(f"Max: {chunk_lengths.max()}")

if __name__ == "__main__":
    main()