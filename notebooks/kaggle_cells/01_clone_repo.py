"""
# CELL 1: Clone Repository

Paste this into the first code cell of your Kaggle notebook.
"""
import os
import sys

REPO_URL = "https://github.com/hasanimam72108/-bangla-handwritten-ocr"
REPO_DIR = "/kaggle/working/bangla-handwritten-ocr"

if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
else:
    # Pull latest changes (tokenizer fix, config update)
    !git -C {REPO_DIR} pull
    
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print(f"Working directory: {os.getcwd()}")
print(f"Contents: {os.listdir('.')}")
