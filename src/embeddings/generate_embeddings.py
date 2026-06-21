import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

def generate_embeddings():
    # Load the chunks data
    print("Loading chunks data...")
    chunks_df = pd.read_parquet('data/processed/cuad_chunks.parquet')
    print(f"Loaded {len(chunks_df)} chunks")
    
    # Initialize the model
    print("Initializing sentence transformer model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("Model initialized successfully")
    
    # Prepare metadata for saving
    metadata_columns = ['chunk_id', 'contract_id', 'clause_type', 'annotation', 'is_risky']
    metadata_df = chunks_df[metadata_columns].copy()
    
    # Generate embeddings in batches
    print("Generating embeddings...")
    batch_size = 32
    embeddings = []
    
    # Use tqdm for progress bar
    for i in tqdm(range(0, len(chunks_df), batch_size), desc="Processing batches"):
        batch = chunks_df.iloc[i:i+batch_size]
        batch_texts = batch['chunk_text'].tolist()
        
        # Generate embeddings for this batch
        batch_embeddings = model.encode(batch_texts)
        embeddings.extend(batch_embeddings)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings)
    print(f"Generated embeddings with shape: {embeddings_array.shape}")
    
    # Save embeddings
    print("Saving embeddings...")
    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/cuad_embeddings.npy', embeddings_array)
    print("Embeddings saved successfully")
    
    # Save metadata
    print("Saving metadata...")
    metadata_df.to_parquet('data/processed/cuad_embedding_metadata.parquet', index=False)
    print("Metadata saved successfully")
    
    print("Embedding generation complete!")

if __name__ == "__main__":
    generate_embeddings()