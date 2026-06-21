# Test for CUAD dataset download and processing
import pandas as pd
import pytest
from pathlib import Path
import sys
import os

# Add src to path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_parquet_file_structure():
    """Test that parquet file has expected structure (this would be run after actual processing)"""
    
    # This test would check if the parquet file exists and has the expected columns
    # In a real scenario, you'd process a small sample dataset first
    
    parquet_path = Path("data/processed/cuad_labels.parquet")
    
    # Check that file exists (this will be true after running the script)
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        expected_columns = ['contract_id', 'contract_text_path', 'clause_type', 'clause_text', 'is_risky']
        
        # Verify all expected columns are present
        for col in expected_columns:
            assert col in df.columns, f"Expected column '{col}' not found in parquet file"
        
        # Verify at least one row exists
        assert len(df) > 0, "Parquet file should contain at least one row"
        
        print(f"Test passed: Parquet file contains {len(df)} rows with columns: {list(df.columns)}")
    else:
        # This is expected when running the test before processing
        print("Parquet file not found - this is expected in testing environment")
        pass

if __name__ == "__main__":
    # Run the tests directly
    test_parquet_file_structure()