"""
# CELL 4: Create Unified labels.csv (no copy — reference originals in-place)

Paste this into the fourth code cell.
Images stay in /kaggle/input/ (quota-free), only paths are stored.
"""
saved_records = []
for i, rec in enumerate(all_records):
    saved_records.append({"image": rec["image_path"], "text": rec["text"]})
    if (i + 1) % 50 == 0:
        print(f"  Indexed {i + 1}/{len(all_records)}")

csv_path = os.path.join(DATA_DIR, "labels.csv")
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "text"])
    writer.writeheader()
    writer.writerows(saved_records)

print(f"Saved {len(saved_records)} records to {csv_path}")
