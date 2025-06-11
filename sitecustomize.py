# sitecustomize.py
# version: 0.1.0
# path: sitecustomize.py

"""Ensure tests can import src modules without modifying PYTHONPATH."""

import os
import sys
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
