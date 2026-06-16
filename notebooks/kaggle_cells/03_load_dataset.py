"""
# CELL 3: Load BN-HTRd Dataset from Hugging Face

Paste this into the third code cell.
"""
import numpy as np
import pandas as pd
from datasets import load_dataset
from pathlib import Path
import csv
import json
import random
from PIL import Image

DATA_DIR = "/kaggle/working/data/bn_htrd"
os.makedirs(DATA_DIR, exist_ok=True)

print("Loading BN-HTRd from Hugging Face...")
dataset = load_dataset("shaoncsecu/BN-HTRd_Splitted", split="train")
print(f"Loaded {len(dataset)} samples")
