"""
# CELL 6: Predict Sample
Paste this into the sixth code cell.
Run this ONLY AFTER training is complete to visually test your model on a random image.
"""
import pandas as pd
import random
import os

val_csv_path = '/kaggle/working/data/val.csv'
if os.path.exists(val_csv_path):
    df = pd.read_csv(val_csv_path)
    sample_image = df.sample(1)['image'].values[0]
    sample_path = os.path.join('/kaggle/working/data/train', sample_image)
    
    !python /kaggle/working/-bangla-handwritten-ocr/predict.py --image {sample_path} --checkpoint /kaggle/working/checkpoints/best_model.pt --config /kaggle/working/-bangla-handwritten-ocr/configs/train_config.yaml --data_dir /kaggle/working/data
else:
    print("Run training first!")
