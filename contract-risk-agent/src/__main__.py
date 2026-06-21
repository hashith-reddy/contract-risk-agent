#!/usr/bin/env python3
"""
Main entry point for the Contract Risk Agent.
This file serves as the primary interface for running the application.
"""

import sys
import os
from pathlib import Path

# Add src to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main entry point for the contract risk agent"""
    print("Contract Risk Agent")
    print("=" * 30)
    print("Available commands:")
    print("- python -m contract_risk_agent download-cuad")
    print("- python -m contract_risk_agent process-contracts") 
    print("- python -m contract_risk_agent run-streamlit")
    print("- python -m contract_risk_agent evaluate")
    
    # This script can be extended to include more functionality
    # For now, we'll just show the available options

if __name__ == "__main__":
    main()