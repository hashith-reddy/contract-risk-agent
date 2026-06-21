# Download and process the CUAD dataset
import argparse
import os
import pandas as pd
import zipfile
import requests
from pathlib import Path
import sys

def download_file(url, filename):
    """Download a file from URL with error handling"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main(limit=None):
    # Create data directories if they don't exist
    raw_data_dir = Path("data/raw/cuad")
    processed_data_dir = Path("data/processed")
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the dataset URL from Zenodo
    dataset_url = "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1"
    
    zip_path = raw_data_dir / "CUAD_v1.zip"
    
    # Download the dataset
    print("Downloading CUAD dataset from Zenodo...")
    if not download_file(dataset_url, zip_path):
        print("Failed to download the dataset")
        return False
    
    # Extract the zip file
    print("Extracting dataset...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
         for member in zip_ref.namelist():
          if member.lower().endswith(".csv"):
            zip_ref.extract(member, raw_data_dir)
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        return False
    
    # Find the master clauses CSV file
    csv_file = None
    for file in raw_data_dir.rglob("*.csv"):
        if "master" in file.name.lower():
            csv_file = file
            break
    
    if not csv_file:
        print("Master clauses CSV file not found")
        return False
    
    # Read the master CSV
    print("Processing master CSV...")

    try:
        df = pd.read_csv(csv_file)

        print("\nColumns:")
        print(df.columns.tolist())

        print("\nFirst Row:")
        print(df.iloc[0])

        print("\nShape:")
        print(df.shape)

        return True

    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False

    
    # Process the data to create the required structure
    processed_rows = []
    
    # The CUAD dataset has a specific structure, we need to extract relevant columns
    # and determine if a clause is risky based on the specified risk categories
    risk_categories = [
        "Anti-Assignment", 
        "Auto-Renewal", 
        "Cap On Liability", 
        "Uncapped Liability",
        "Non-Compete", 
        "Most Favored Nation"
    ]
    
    # Process each row in the dataframe
    for _, row in df.iterrows():
        if limit and len(processed_rows) >= limit:
            break
            
        # Get contract ID and text path
        contract_id = row.get('contract_id', '')
        clause_text = row.get('clause_text', '')
        clause_type = row.get('clause_type', '')
        
        # Determine if the clause is risky based on risk categories
        is_risky = 0
        
        # Check each risk category - if any of them have non-empty answers, mark as risky
        for category in risk_categories:
            answer = row.get(category, '')
            if pd.notna(answer) and str(answer).strip():
                is_risky = 1
                break
        
        # Get the contract text path (this will depend on how the dataset is structured)
        contract_text_path = ''
        
        # Try to find a path based on contract ID or other identifiers
        if 'contract_id' in row:
            contract_text_path = f"contracts/{row['contract_id']}.txt"
        
        processed_rows.append({
            'contract_id': contract_id,
            'contract_text_path': contract_text_path,
            'clause_type': clause_type,
            'clause_text': clause_text,
            'is_risky': is_risky
        })
    
    # Create a DataFrame with the processed data
    result_df = pd.DataFrame(processed_rows)
    
    # Save to parquet file
    output_path = processed_data_dir / "cuad_labels.parquet"
    try:
        result_df.to_parquet(output_path, index=False)
        print(f"Processed {len(result_df)} rows and saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving parquet file: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process CUAD dataset")
    parser.add_argument("--limit", type=int, help="Limit processing to N contracts for fast iteration")
    
    args = parser.parse_args()
    
    success = main(args.limit)
    sys.exit(0 if success else 1)