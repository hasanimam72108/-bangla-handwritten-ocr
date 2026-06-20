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
!ln -sf /kaggle/working/data/train /kaggle/working/data/val

!python train.py --config configs/train_config.yaml --data_dir /kaggle/working/data --output_dir /kaggle/working/checkpoints
