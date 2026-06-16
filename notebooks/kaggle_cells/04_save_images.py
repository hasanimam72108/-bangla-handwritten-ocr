"""
# CELL 4: Save Images and Create Unified labels.csv

Paste this into the fourth code cell.
"""
image_dir = os.path.join(DATA_DIR, "images")
os.makedirs(image_dir, exist_ok=True)

saved_records = []
for i, rec in enumerate(all_records):
    ext = Path(rec["image_path"]).suffix
    img_filename = f"img_{i:05d}{ext}"
    img_dst = os.path.join(image_dir, img_filename)

    img = Image.open(rec["image_path"])
    img = img.convert("RGB")
    img.save(img_dst, "JPEG", quality=95)

    saved_records.append({"image": img_filename, "text": rec["text"]})

    if (i + 1) % 50 == 0:
        print(f"  Copied {i + 1}/{len(all_records)}")

csv_path = os.path.join(DATA_DIR, "labels.csv")
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "text"])
    writer.writeheader()
    writer.writerows(saved_records)

print(f"Saved {len(saved_records)} records to {csv_path}")
