import os
import glob
import pandas as pd
import shutil
from tqdm import tqdm

def prepare_data():
    # Correct base path for the Kaggle dataset
    base_dir = "/kaggle/input/datasets/jawadurrafid/bn-htrd-dataset/BN-HTRd A Benchmark Dataset for Document Level Offline Bangla Handwritten Text Recognition (HTR)/BN-HTR_Dataset/BN-HTR_Dataset"
    
    text_dir = os.path.join(base_dir, "Recognition_Ground_Truth_Texts")
    lines_dir = os.path.join(base_dir, "Segmentation_Images", "Lines")
    
    output_dir = "/kaggle/working/data"
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    
    data = []
    
    # 1. Iterate over each document folder (e.g. '7', '135')
    doc_folders = [f for f in os.listdir(text_dir) if os.path.isdir(os.path.join(text_dir, f))]
    
    print(f"Found {len(doc_folders)} document folders. Processing...")
    
    for doc_id in tqdm(doc_folders):
        txt_path = os.path.join(text_dir, doc_id, f"{doc_id}.txt")
        if not os.path.exists(txt_path):
            continue
            
        # Read all text lines for this document
        with open(txt_path, 'r', encoding='utf-8') as f:
            # Drop empty lines at the end but keep formatting
            text_lines = [line.strip() for line in f.read().split('\n') if line.strip()]
            
        # Now look for the segmented images for this document
        doc_lines_dir = os.path.join(lines_dir, doc_id)
        if not os.path.exists(doc_lines_dir):
            continue
            
        # Find writer folders like '7_1', '7_2' inside '7'
        writer_folders = [f for f in os.listdir(doc_lines_dir) if os.path.isdir(os.path.join(doc_lines_dir, f))]
        
        for writer_id in writer_folders:
            writer_dir = os.path.join(doc_lines_dir, writer_id)
            
            # For each writer, they should have cropped images like '7_1_1.jpg', '7_1_2.jpg'
            # We match the integer suffix to the text_lines index
            images = glob.glob(os.path.join(writer_dir, "*.*"))
            image_files = [img for img in images if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            for img_path in image_files:
                img_name = os.path.basename(img_path) # e.g., '7_1_3.jpg'
                base_name = os.path.splitext(img_name)[0] # e.g., '7_1_3'
                
                try:
                    # Extract the line number (the last part after the last underscore)
                    line_idx_str = base_name.split('_')[-1]
                    line_idx = int(line_idx_str) - 1 # 0-indexed for our array
                    
                    if 0 <= line_idx < len(text_lines):
                        text = text_lines[line_idx]
                        
                        if len(text) > 0:
                            # Copy image to Kaggle working dir
                            dest_path = os.path.join(output_dir, "train", img_name)
                            shutil.copy2(img_path, dest_path)
                            
                            data.append({
                                "image": img_name,
                                "text": text
                            })
                except Exception as e:
                    pass # Skip if filename format is unexpected

    from sklearn.model_selection import train_test_split
    df = pd.DataFrame(data)
    
    # Split 90% train, 10% validation
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    train_csv_path = os.path.join(output_dir, "train.csv")
    val_csv_path = os.path.join(output_dir, "val.csv")
    
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    
    print(f"\nSuccessfully aligned and mapped {len(df)} perfectly cropped line images to text!")
    print(f"Saved {len(train_df)} items to {train_csv_path}")
    print(f"Saved {len(val_df)} items to {val_csv_path}")
    print(train_df.head())

if __name__ == "__main__":
    prepare_data()
