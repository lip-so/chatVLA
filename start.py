#!/usr/bin/env python3
"""
Start script - alias for main.py
This file exists for backwards compatibility with Dockerfiles that reference start.py
"""

import os
import sys

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.insert(0, os.path.dirname(__file__))

# Import and run the main application
if __name__ == '__main__':
    from main import *