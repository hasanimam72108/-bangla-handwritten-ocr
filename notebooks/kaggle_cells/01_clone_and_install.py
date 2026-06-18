"""
# CELL 1: Clone Repository & Install Dependencies
Paste this into the first code cell of your Kaggle notebook.
"""
import os
import sys

REPO_URL = "https://github.com/hasanimam72108/-bangla-handwritten-ocr"
REPO_DIR = "/kaggle/working/-bangla-handwritten-ocr"

if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
else:
    !git -C {REPO_DIR} pull
    
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

!pip install -r requirements.txt

print(f"Working directory: {os.getcwd()}")
print("Setup complete!")
