import os
import glob
import pandas as pd
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def prepare_data():
    base_dir = "/kaggle/input/datasets/jawadurrafid/bn-htrd-dataset/BN-HTRd A Benchmark Dataset for Document Level Offline Bangla Handwritten Text Recognition (HTR)/BN-HTR_Dataset/BN-HTR_Dataset"
    
    text_dir = os.path.join(base_dir, "Recognition_Ground_Truth_Texts")
    lines_dir = os.path.join(base_dir, "Segmentation_Images", "Lines")
    
    output_dir = "/kaggle/working/data"
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    
    data = []
    
    doc_folders = [f for f in os.listdir(text_dir) if os.path.isdir(os.path.join(text_dir, f))]
    print(f"Found {len(doc_folders)} document folders. Processing using the Ground Truth Excel files...")
    
    for doc_id in tqdm(doc_folders):
        xlsx_path = os.path.join(text_dir, doc_id, f"{doc_id}.xlsx")
        if not os.path.exists(xlsx_path):
            continue
            
        try:
            df = pd.read_excel(xlsx_path)
        except Exception:
            continue
            
        if 'Id' not in df.columns or 'Word' not in df.columns:
            continue
            
        # Drop missing values
        df = df.dropna(subset=['Id', 'Word'])
        df['Id'] = df['Id'].astype(str)
        df['Word'] = df['Word'].astype(str)
        
        # The 'Id' is format: doc_writer_line_word (e.g. 7_1_1_1)
        # We want to group by line: doc_writer_line (e.g. 7_1_1)
        # We split by '_' from the right exactly once to remove the word number.
        df['line_id'] = df['Id'].apply(lambda x: x.rsplit('_', 1)[0])
        
        # Group by line and join the words with a space!
        # sort=False ensures we keep the exact order of words as they appear in the Excel sheet
        line_df = df.groupby('line_id', sort=False)['Word'].apply(lambda x: ' '.join(x)).reset_index()
        
        for _, row in line_df.iterrows():
            line_id = row['line_id']
            text = row['Word'].strip()
            
            if not text:
                continue
                
            # Reconstruct the image path
            # line_id is like '7_1_1' -> doc_id='7', writer_id='7_1'
            parts = line_id.split('_')
            if len(parts) < 3:
                continue
                
            current_doc = parts[0]
            writer_id = f"{parts[0]}_{parts[1]}"
            
            img_dir = os.path.join(lines_dir, current_doc, writer_id)
            
            # Look for the exact image file
            found_img = None
            for ext in ['.jpg', '.png', '.jpeg']:
                potential = os.path.join(img_dir, f"{line_id}{ext}")
                if os.path.exists(potential):
                    found_img = potential
                    break
                    
            if found_img:
                dest_name = f"{line_id}.jpg"
                dest_path = os.path.join(output_dir, "train", dest_name)
                shutil.copy2(found_img, dest_path)
                data.append({
                    "image": dest_name,
                    "text": text
                })
                
    final_df = pd.DataFrame(data)
    
    # Split 90% train, 10% validation
    train_df, val_df = train_test_split(final_df, test_size=0.1, random_state=42)
    
    train_csv_path = os.path.join(output_dir, "train.csv")
    val_csv_path = os.path.join(output_dir, "val.csv")
    
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    
    print(f"\nSuccessfully mapped {len(final_df)} perfectly aligned line images!")
    print(f"Saved {len(train_df)} train items to {train_csv_path}")
    print(f"Saved {len(val_df)} val items to {val_csv_path}")
    print(train_df.head())

if __name__ == "__main__":
    prepare_data()
