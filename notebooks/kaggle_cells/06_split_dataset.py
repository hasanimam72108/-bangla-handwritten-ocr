"""
# CELL 6: Split Dataset into Train/Val/Test

Paste this into the sixth code cell.
Images stay at their original Kaggle input paths (no copying).
"""
random.seed(42)

indices = list(range(len(df)))
random.shuffle(indices)

n = len(indices)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

def save_split(indices, split_name):
    rows = []
    for idx in indices:
        row = df.iloc[idx]
        rows.append({"image": row["image"], "text": row["text"]})
    out_csv = os.path.join(DATA_DIR, f"{split_name}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    return len(rows)

print("Splitting dataset:")
print(f"  Train: {save_split(train_idx, 'train')}")
print(f"  Val:   {save_split(val_idx, 'val')}")
print(f"  Test:  {save_split(test_idx, 'test')}")
