"""
# CELL 4: Train Model
Paste this into the fourth code cell.
This automatically ensures your dataset is split into train/val and then starts training!
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

train_csv_path = '/kaggle/working/data/train.csv'
val_csv_path = '/kaggle/working/data/val.csv'

# Auto-fix: Split into train/val if val.csv doesn't exist yet
if os.path.exists(train_csv_path) and not os.path.exists(val_csv_path):
    print("Splitting dataset into train/val...")
    df = pd.read_csv(train_csv_path)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    print(f"Created train.csv ({len(train_df)}) and val.csv ({len(val_df)})")

# Fix the validation image directory path!
val_dir = "/kaggle/working/data/val"
train_dir = "/kaggle/working/data/train"
if not os.path.exists(val_dir) and os.path.exists(train_dir):
    os.symlink(train_dir, val_dir)

# Automatically resume from the most recent model if it exists!
resume_arg = ""
if os.path.exists("/kaggle/working/checkpoints/last_model.pt"):
    resume_arg = "--resume /kaggle/working/checkpoints/last_model.pt"
elif os.path.exists("/kaggle/working/checkpoints/best_model.pt"):
    resume_arg = "--resume /kaggle/working/checkpoints/best_model.pt"

!python /kaggle/working/-bangla-handwritten-ocr/train.py --config /kaggle/working/-bangla-handwritten-ocr/configs/train_config.yaml --data_dir /kaggle/working/data --output_dir /kaggle/working/checkpoints {resume_arg}
