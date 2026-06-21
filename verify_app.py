#!/usr/bin/env python3
"""
Verification script for the Gradio app.
"""

import sys
import os

# Add the project root to the path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required imports work."""
    try:
        import gradio as gr
        print("✓ Gradio imported successfully")
        
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
        print("✓ All analysis functions imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {str(e)}")
        return False

def test_data_files():
    """Test that required data files exist."""
    try:
        embeddings_path = 'data/processed/cuad_embeddings.npy'
        metadata_path = 'data/processed/cuad_embedding_metadata.parquet'
        chunks_path = 'data/processed/cuad_chunks.parquet'
        
        files_exist = [
            os.path.exists(embeddings_path),
            os.path.exists(metadata_path), 
            os.path.exists(chunks_path)
        ]
        
        if all(files_exist):
            print("✓ All required data files found")
            return True
        else:
            print("✗ Some data files missing:")
            for path, exists in zip([embeddings_path, metadata_path, chunks_path], files_exist):
                print(f"  {path}: {'Found' if exists else 'Missing'}")
            return False
            
    except Exception as e:
        print(f"✗ Data file check error: {str(e)}")
        return False

def test_basic_functionality():
    """Test that basic functionality works."""
    try:
        from src.risk_analysis.analyze_contract import generate_risk_report
        result = generate_risk_report("Test contract text", [])
        print("✓ Basic analysis function works")
        return True
    except Exception as e:
        print(f"✗ Basic functionality error: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== Contract Risk Analysis App Verification ===\n")
    
    tests = [
        ("Import Check", test_imports),
        ("Data Files Check", test_data_files), 
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        result = test_func()
        results.append(result)
        print()
    
    if all(results):
        print("🎉 ALL TESTS PASSED - App is ready to run!")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)