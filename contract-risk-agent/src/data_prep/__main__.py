#!/usr/bin/env python3
"""
Main entry point for data preparation scripts.
This file allows running data preparation tasks using `python -m data_prep`
"""

import sys
import os
from pathlib import Path

# Add src to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    """Main entry point for data preparation"""
    print("Data preparation tools:")
    print("- python -m data_prep.download_cuad [--limit N]")
    print("- python -m data_prep.process_contracts")
    
    # This script can be extended to include more data preparation tasks
    # For now, we'll just show the available options

if __name__ == "__main__":
    main()