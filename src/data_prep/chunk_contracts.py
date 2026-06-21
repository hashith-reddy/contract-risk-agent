import pandas as pd
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter

def generate_chunk_id(contract_id, clause_type, chunk_text):
    """Generate deterministic chunk ID based on contract_id, clause_type and chunk_text"""
    # Create a unique identifier for the chunk
    chunk_key = f"{contract_id}_{clause_type}_{chunk_text[:100]}"  # Using first 100 chars to keep it manageable
    chunk_hash = hashlib.md5(chunk_key.encode()).hexdigest()
    return chunk_hash

def chunk_contracts(input_file, output_file):
    """Chunk contracts using RecursiveCharacterTextSplitter"""
    
    # Read the input data
    df = pd.read_parquet(input_file)
    
    # Initialize the text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    
    for _, row in df.iterrows():
        contract_id = row['contract_id']
        clause_type = row['clause_type']
        clause_text = row['clause_text']
        annotation = row['annotation']
        is_risky = row['is_risky']
        
        # Split the clause text into chunks
        chunk_texts = splitter.split_text(clause_text)
        
        # Create chunks with metadata
        for chunk_text in chunk_texts:
            chunk_id = generate_chunk_id(contract_id, clause_type, chunk_text)
            
            chunks.append({
                'chunk_id': chunk_id,
                'contract_id': contract_id,
                'clause_type': clause_type,
                'chunk_text': chunk_text,
                'annotation': annotation,
                'is_risky': is_risky
            })
    
    # Create DataFrame from chunks
    chunks_df = pd.DataFrame(chunks)
    
    # Save to parquet
    chunks_df.to_parquet(output_file, index=False)
    
    print(f"Processed {len(df)} contracts into {len(chunks_df)} chunks")
    return chunks_df

if __name__ == "__main__":
    input_file = "data/processed/cuad_labels.parquet"
    output_file = "data/processed/cuad_chunks.parquet"
    chunk_contracts(input_file, output_file)