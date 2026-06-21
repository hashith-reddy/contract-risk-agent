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
    
    # Read the master CSV directly from the provided path
    csv_file = Path("C:/Users/AMD/Downloads/CUAD_v1/CUAD_v1/master_clauses.csv")
    
    if not csv_file.exists():
        print("Master clauses CSV file not found at expected location")
        return False
    
    # Read the master CSV
    try:
        df = pd.read_csv(csv_file)
        
        # Process the data to create the required long-format structure
        processed_rows = []
        
        # Define risk categories as specified in requirements
        risk_categories = [
            "Most Favored Nation",
            "Non-Compete", 
            "Anti-Assignment",
            "Cap On Liability", 
            "Uncapped Liability",
            "Change Of Control",
            "Termination For Convenience",
            "Renewal Term"
        ]
        
        # Process each row in the dataframe (contract)
        for idx, row in df.iterrows():
            if limit and idx >= limit:
                break
                
            # Get contract ID from Filename column
            contract_id = row.get('Filename', '')
            
            # Process each clause category
            for clause_type in risk_categories:
                # Skip if the clause column doesn't exist or is empty
                clause_column = clause_type
                answer_column = f"{clause_type}-Answer"
                
                # Get clause text (from the main clause column)
                clause_text = row.get(clause_column, '')
                
                # Get annotation value (from the -Answer column)
                annotation = row.get(answer_column, '')
                
                # Determine if the clause is risky based on risk categories
                # A clause is risky if it has an annotation value that indicates a risk
                is_risky = 0
                
                # If there's an annotation and it's not empty or "No" or "[]", mark as risky
                if pd.notna(annotation) and str(annotation).strip() and str(annotation).strip().lower() != 'no' and str(annotation).strip() != '[]':
                    is_risky = 1
                
                # Skip empty clause texts
                if clause_text in ["", "[]", "nan"]:
                    continue

                # Only add rows where there's actual clause text (not empty)
                if pd.notna(clause_text) and str(clause_text).strip():
                    processed_rows.append({
                        'contract_id': contract_id,
                        'clause_type': clause_type,
                        'clause_text': str(clause_text),
                        'annotation': str(annotation) if pd.notna(annotation) else '',
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
            
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process CUAD dataset")
    parser.add_argument("--limit", type=int, help="Limit processing to N contracts for fast iteration")
    
    args = parser.parse_args()
    
    success = main(args.limit)
    sys.exit(0 if success else 1)
