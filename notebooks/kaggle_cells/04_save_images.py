"""
# CELL 4: Save Images and Create labels.csv

Paste this into the fourth code cell.
"""
image_dir = os.path.join(DATA_DIR, "images")
os.makedirs(image_dir, exist_ok=True)

records = []
for i, sample in enumerate(dataset):
    image = sample["image"]
    text = sample["text"]
    
    img_filename = f"img_{i:05d}.jpg"
    img_path = os.path.join(image_dir, img_filename)
    
    if isinstance(image, Image.Image):
        image.save(img_path, "JPEG", quality=95)
    else:
        Image.fromarray(np.array(image)).save(img_path, "JPEG", quality=95)
    
    records.append({"image": img_filename, "text": text})
    
    if (i + 1) % 200 == 0:
        print(f"  Processed {i + 1}/{len(dataset)}")

csv_path = os.path.join(DATA_DIR, "labels.csv")
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "text"])
    writer.writeheader()
    writer.writerows(records)
print(f"Saved {len(records)} images and labels")
