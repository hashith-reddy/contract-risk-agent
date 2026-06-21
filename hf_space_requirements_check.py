#!/usr/bin/env python3
"""
Verification script to check if all required files exist for Hugging Face Spaces deployment.
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    exists = os.path.exists(filepath)
    status = "PASS" if exists else "FAIL"
    print(f"{status}: {description} ({filepath})")
    return exists

def main():
    print("=== Hugging Face Spaces Requirements Check ===\n")
    
    required_files = [
        ("app.py", "Main Gradio application file"),
        ("data/processed/cuad_chunks.parquet", "Contract chunks data file"),
        ("data/processed/cuad_embeddings.npy", "Embeddings data file"),
        ("data/processed/cuad_embedding_metadata.parquet", "Embedding metadata file")
    ]
    
    all_passed = True
    
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("✅ ALL REQUIREMENTS PASSED - Ready for Hugging Face deployment")
        sys.exit(0)
    else:
        print("❌ SOME REQUIREMENTS FAILED - Please fix the missing files")
        sys.exit(1)

if __name__ == "__main__":
    main()