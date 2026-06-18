import os
import glob
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def segment_lines(image_path, text_content):
    """
    Given an image path and its full text, attempt to crop lines.
    Returns a list of cropped image numpy arrays and corresponding text lines.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [], []
    
    # Binarize
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Horizontal projection profile
    proj = np.sum(thresh, axis=1)
    
    # Smooth the projection to find distinct lines
    kernel_size = max(5, img.shape[0] // 100)
    kernel = np.ones(kernel_size) / kernel_size
    proj_smoothed = np.convolve(proj, kernel, mode='same')
    
    # Threshold for finding text lines
    proj_threshold = np.max(proj_smoothed) * 0.1
    is_text = proj_smoothed > proj_threshold
    
    lines = []
    start_idx = -1
    for i in range(len(is_text)):
        if is_text[i] and start_idx == -1:
            start_idx = i
        elif not is_text[i] and start_idx != -1:
            # End of a line segment
            if i - start_idx > 10: # minimum pixel height of a line
                lines.append((max(0, start_idx - 10), min(img.shape[0], i + 10)))
            start_idx = -1
            
    if start_idx != -1 and img.shape[0] - start_idx > 10:
        lines.append((max(0, start_idx - 10), img.shape[0]))
        
    # Split text into lines/sentences
    # We split by '।' (dari), '?', '!', or newlines.
    import re
    text_content = text_content.strip()
    raw_text_lines = re.split(r'[\n]+', text_content)
    text_lines = []
    for rl in raw_text_lines:
        sub_lines = [s.strip() for s in re.split(r'(?<=[।?!])\s+', rl) if s.strip()]
        if not sub_lines:
            text_lines.append(rl.strip())
        else:
            text_lines.extend(sub_lines)
            
    text_lines = [t for t in text_lines if len(t) > 2]
    
    # Match cropped lines to text_lines.
    cropped_imgs = []
    mapped_texts = []
    
    # Map 1-to-1 up to the minimum length
    min_len = min(len(lines), len(text_lines))
    if min_len == 0:
        return [], []
        
    img_color = cv2.imread(image_path)
    for i in range(min_len):
        y1, y2 = lines[i]
        crop = img_color[y1:y2, :]
        cropped_imgs.append(crop)
        mapped_texts.append(text_lines[i])
        
    return cropped_imgs, mapped_texts


def main():
    # Kaggle input paths
    datasets = [
        "/kaggle/input/banglawriting-with-page-level-annotations/Annotations",
        "/kaggle/input/handwritten-text-recognition-bongabdo/Bongabdo1429/Annotations",
        # Adding a fallback path for Kaggle structure variances
        "/kaggle/input/datasets/ayanwap7/banglawriting-with-page-level-annotations/Annotations",
        "/kaggle/input/datasets/joebeachcapital/handwritten-text-recognition-bongabdo/Bongabdo1429/Annotations"
    ]
    
    output_dir = "/kaggle/working/data"
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    
    all_data = []
    img_extensions = ['.jpg', '.jpeg', '.png']
    global_idx = 0
    
    for ann_dir in datasets:
        if not os.path.exists(ann_dir):
            continue
            
        print(f"Processing directory: {ann_dir}")
        img_dir = ann_dir.replace("Annotations", "Images")
            
        txt_files = glob.glob(os.path.join(ann_dir, "*.txt"))
        
        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
                
            base_name = os.path.splitext(os.path.basename(txt_file))[0]
            
            # Find matching image
            img_path = None
            for ext in img_extensions:
                cand = os.path.join(img_dir, base_name + ext)
                if os.path.exists(cand):
                    img_path = cand
                    break
            
            # For bongabdo, sometimes the image has a suffix 'a' or 'b'
            if img_path is None:
                cand_files = glob.glob(os.path.join(img_dir, base_name + "*.*"))
                if cand_files:
                    img_path = cand_files[0]
                    
            if img_path and os.path.exists(img_path):
                crops, texts = segment_lines(img_path, text)
                for crop, t in zip(crops, texts):
                    # Save some minimum dimensions
                    if crop.shape[0] > 10 and crop.shape[1] > 10:
                        all_data.append({
                            "crop_img": crop,
                            "text": t,
                            "filename": f"crop_{global_idx}.jpg"
                        })
                        global_idx += 1
                    
    print(f"Total extracted line pairs: {len(all_data)}")
    
    if len(all_data) == 0:
        print("No data processed. Ensure paths are correct when running on Kaggle.")
        # Create a dummy CSV so it doesn't fail completely if testing locally
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(columns=["image", "text"]).to_csv(os.path.join(output_dir, "train.csv"), index=False)
        pd.DataFrame(columns=["image", "text"]).to_csv(os.path.join(output_dir, "val.csv"), index=False)
        return
        
    train_data, val_data = train_test_split(all_data, test_size=0.15, random_state=42)
    
    def save_split(data_list, split_name):
        records = []
        split_dir = os.path.join(output_dir, split_name)
        for item in data_list:
            out_path = os.path.join(split_dir, item["filename"])
            cv2.imwrite(out_path, item["crop_img"])
            records.append({
                "image": item["filename"],
                "text": item["text"]
            })
        df = pd.DataFrame(records)
        df.to_csv(os.path.join(output_dir, f"{split_name}.csv"), index=False)
        print(f"Saved {split_name}.csv with {len(df)} records")
        
    save_split(train_data, "train")
    save_split(val_data, "val")

if __name__ == "__main__":
    main()
