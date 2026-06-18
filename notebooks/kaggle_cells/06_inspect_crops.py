"""
# CELL 6: Inspect Cropped Images
Create a new code cell in Kaggle and paste this code to visually inspect how well the auto-cropper worked.
"""
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load the training CSV
train_csv_path = "/kaggle/working/data/train.csv"
if not os.path.exists(train_csv_path):
    print("Data not found. Did you run the prepare_data cell?")
else:
    df = pd.read_csv(train_csv_path)
    
    # Pick 5 random samples
    samples = df.sample(5)
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 10))
    fig.tight_layout(pad=3.0)
    
    for i, (_, row) in enumerate(samples.iterrows()):
        img_path = os.path.join("/kaggle/working/data/train", row['image'])
        text = row['text']
        
        try:
            img = Image.open(img_path)
            axes[i].imshow(img, cmap='gray' if img.mode == 'L' else None)
            axes[i].set_title(text, fontdict={'fontsize': 14}, loc='left')
            axes[i].axis('off')
        except Exception as e:
            axes[i].set_title(f"Error loading {row['image']}: {e}", color="red")
            axes[i].axis('off')
            
    plt.show()
