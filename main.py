#!/usr/bin/env python3
"""
Syndgen Main Entry Point

Main script for running the Syndgen synthetic data generation pipeline.
"""

import sys
import os

# Add the current directory to Python path to ensure syndgen package is found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the CLI function directly
from cli import main

if __name__ == "__main__":
    main()
