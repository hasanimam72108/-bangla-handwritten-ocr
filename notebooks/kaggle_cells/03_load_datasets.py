"""
# CELL 3: Load Bongabdo & BanglaWriting from Kaggle Inputs

Scans Kaggle input directories, loads images + annotations from both datasets,
and creates labels.csv.

Paste this into the third code cell.
"""
import os, sys, csv, json, random
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from glob import glob

DATA_DIR = "/kaggle/working/data/bangla_combined"
os.makedirs(DATA_DIR, exist_ok=True)

# --- Kaggle input paths (auto-discover) ---
# Try both nested (datasets/) and flat paths
KAGGLE_CANDIDATES = ["/kaggle/input/datasets", "/kaggle/input"]
KAGGLE_INPUT = None
for path in KAGGLE_CANDIDATES:
    if os.path.isdir(path):
        entries = [d for d in os.listdir(path) if d not in (".ipynb_checkpoints", "__pycache__")]
        if entries:
            KAGGLE_INPUT = path
            break
if not KAGGLE_INPUT:
    raise RuntimeError("Could not find Kaggle input directory")

print(f"Kaggle input base: {KAGGLE_INPUT}")
print("Available inputs:")
for d in sorted(os.listdir(KAGGLE_INPUT)):
    print(f"  {KAGGLE_INPUT}/{d}/")

# Find Bongabdo and BanglaWriting paths
BONGABDO_PATH = None
BANGLAWRITING_PATH = None
for root, dirs, files in os.walk(KAGGLE_INPUT):
    if "Images" in dirs and "Annotations" in dirs and not BONGABDO_PATH and "bongabdo" in root.lower():
        BONGABDO_PATH = root
    if "Images" in dirs and "Annotations" in dirs and not BANGLAWRITING_PATH and "banglawriting" in root.lower():
        BANGLAWRITING_PATH = root

if not BONGABDO_PATH:
    for d in os.listdir(KAGGLE_INPUT):
        full = os.path.join(KAGGLE_INPUT, d)
        if "bongabdo" in d.lower():
            BONGABDO_PATH = full
if not BANGLAWRITING_PATH:
    for d in os.listdir(KAGGLE_INPUT):
        full = os.path.join(KAGGLE_INPUT, d)
        if "banglawriting" in d.lower() or "bangla-writing" in d.lower() or "bangla_writing" in d.lower():
            BANGLAWRITING_PATH = full

print(f"\nBongabdo root: {BONGABDO_PATH}")
print(f"BanglaWriting root: {BANGLAWRITING_PATH}")

# --- Scan directory for images and annotations ---
def scan_dataset(base_path):
    """Recursively find all images + corresponding annotations."""
    image_exts = {'.jpg', '.jpeg', '.png'}
    records = []

    # Walk all subdirectories
    for root, dirs, files in os.walk(base_path):
        # Find image files
        img_files = [f for f in files if Path(f).suffix.lower() in image_exts]
        for f in img_files:
            img_path = os.path.join(root, f)
            stem = Path(f).stem

            # Look for matching annotation file (same stem)
            # Try: .txt, .json in same dir, or Annotations/ subdir
            ann_text = None

            # Pattern 1: same directory .txt
            txt_candidate = os.path.join(root, stem + ".txt")
            if os.path.exists(txt_candidate):
                with open(txt_candidate, "r", encoding="utf-8") as af:
                    ann_text = af.read().strip()
            
            # Pattern 2: same directory .json
            if ann_text is None:
                json_candidate = os.path.join(root, stem + ".json")
                if os.path.exists(json_candidate):
                    with open(json_candidate, "r", encoding="utf-8") as af:
                        data = json.load(af)
                    # Try to extract page-level text from JSON
                    if "text" in data:
                        ann_text = data["text"]
                    elif "shapes" in data:
                        # BanglaWriting format: concatenate all word labels
                        words = [s["label"] for s in data["shapes"] if s.get("label")]
                        ann_text = " ".join(words)
                    elif "annotations" in data:
                        ann_text = data["annotations"]

            # Pattern 3: Annotations/ subdirectory
            if ann_text is None:
                parent = Path(root).parent
                ann_dir = os.path.join(parent, "Annotations")
                if os.path.isdir(ann_dir):
                    for ext in [".txt", ".json"]:
                        ann_candidate = os.path.join(ann_dir, stem + ext)
                        if os.path.exists(ann_candidate):
                            with open(ann_candidate, "r", encoding="utf-8") as af:
                                content = af.read().strip()
                            if ext == ".json":
                                data = json.loads(content)
                                if "text" in data:
                                    ann_text = data["text"]
                                elif "shapes" in data:
                                    words = [s["label"] for s in data["shapes"] if s.get("label")]
                                    ann_text = " ".join(words)
                                else:
                                    ann_text = str(data)
                            else:
                                ann_text = content
                            break

            if ann_text:
                records.append({"image_path": img_path, "text": ann_text})
    
    return records

print("\nScanning Bongabdo...")
bongabdo_records = scan_dataset(BONGABDO_PATH) if BONGABDO_PATH else []
print(f"  Found {len(bongabdo_records)} samples")

print("Scanning BanglaWriting...")
banglawriting_records = scan_dataset(BANGLAWRITING_PATH) if BANGLAWRITING_PATH else []
print(f"  Found {len(banglawriting_records)} samples")

all_records = bongabdo_records + banglawriting_records
print(f"\nTotal combined samples: {len(all_records)}")

if len(all_records) == 0:
    print("\nDIAGNOSTIC: No samples found.")
    print(f"  BONGABDO_PATH exists: {BONGABDO_PATH and os.path.isdir(BONGABDO_PATH)}")
    print(f"  BANGLAWRITING_PATH exists: {BANGLAWRITING_PATH and os.path.isdir(BANGLAWRITING_PATH)}")
    if BONGABDO_PATH:
        print(f"  Bongabdo contents: {os.listdir(BONGABDO_PATH)[:10]}")
    if BANGLAWRITING_PATH:
        print(f"  BanglaWriting contents: {os.listdir(BANGLAWRITING_PATH)[:10]}")
    raise RuntimeError("No samples found! Check Kaggle dataset input paths.")
