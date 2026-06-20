"""
# CELL 5: Evaluate Model
Paste this into the fifth code cell.
Run this ONLY AFTER training is complete to evaluate your final model metrics.
"""
import os
val_dir = "/kaggle/working/data/val"
train_dir = "/kaggle/working/data/train"
if not os.path.exists(val_dir) and os.path.exists(train_dir):
    os.symlink(train_dir, val_dir)

!pip install jiwer -q
!python /kaggle/working/-bangla-handwritten-ocr/evaluate.py --config /kaggle/working/-bangla-handwritten-ocr/configs/train_config.yaml --data_dir /kaggle/working/data --checkpoint /kaggle/working/checkpoints/best_model.pt --split val
