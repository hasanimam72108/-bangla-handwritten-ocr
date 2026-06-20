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
            
        # Find all page text files (e.g., '7_1.txt', '7_2.txt') to count boxes
        page_txt_files = glob.glob(os.path.join(doc_lines_dir, f"{doc_id}_*.txt"))
        
        # Sort them numerically by page number: 7_1.txt, 7_2.txt ...
        def get_page_num(filepath):
            filename = os.path.basename(filepath)
            name = os.path.splitext(filename)[0] # '7_1'
            return int(name.split('_')[-1])
            
        page_txt_files = sorted(page_txt_files, key=get_page_num)
        
        current_text_offset = 0
        
        for page_txt_path in page_txt_files:
            page_name = os.path.splitext(os.path.basename(page_txt_path))[0] # '7_1'
            
            # Count how many YOLO bounding boxes are in this page
            try:
                with open(page_txt_path, 'r', encoding='utf-8') as pf:
                    num_boxes = sum(1 for line in pf if line.strip())
            except Exception:
                continue
                
            page_folder = os.path.join(doc_lines_dir, page_name)
            if not os.path.isdir(page_folder):
                current_text_offset += num_boxes
                continue
                
            # Now find all images in this page's folder
            images = glob.glob(os.path.join(page_folder, "*.*"))
            image_files = [img for img in images if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            for img_path in image_files:
                img_name = os.path.basename(img_path) # e.g., '7_1_3.jpg'
                base_name = os.path.splitext(img_name)[0] # e.g., '7_1_3'
                
                try:
                    # Extract the line number (the last part after the last underscore)
                    line_idx_str = base_name.split('_')[-1]
                    k = int(line_idx_str) # 1-indexed line number in this specific page
                    
                    # Global text index is the offset + (k - 1)
                    global_text_idx = current_text_offset + (k - 1)
                    
                    if 0 <= global_text_idx < len(text_lines):
                        text = text_lines[global_text_idx]
                        
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
                    
            # Advance the global offset by the number of boxes in this page
            current_text_offset += num_boxes

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
