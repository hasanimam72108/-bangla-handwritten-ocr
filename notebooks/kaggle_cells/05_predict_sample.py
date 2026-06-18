"""
# CELL 5: Test on a random sample
Paste this into the fifth code cell.
This shows a quick visual prediction from the validation set.
"""
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# Load the validation CSV we generated
val_df = pd.read_csv("/kaggle/working/data/val.csv")
sample = val_df.sample(1).iloc[0]

img_path = os.path.join("/kaggle/working/data/val", sample["image"])
ground_truth = sample["text"]

print(f"Ground Truth: {ground_truth}")

# Run prediction script
!python predict.py --image {img_path} --checkpoint /kaggle/working/checkpoints/best_model.pt --tokenizer bng --data_dir /kaggle/working/data

# Display image
img = Image.open(img_path)
plt.figure(figsize=(10, 2))
plt.imshow(img, cmap='gray' if img.mode == 'L' else None)
plt.axis('off')
plt.show()
