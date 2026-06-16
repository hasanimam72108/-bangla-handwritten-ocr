import os
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image


def download_bn_htrd(output_dir: str = "data/bn_htrd"):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading BN-HTRd dataset from Hugging Face...")
    dataset = load_dataset("shaoncsecu/BN-HTRd_Splitted", split="train")
    print(f"Loaded {len(dataset)} samples")

    image_dir = os.path.join(output_dir, "images")
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
            image_pil = Image.fromarray(np.array(image))
            image_pil.save(img_path, "JPEG", quality=95)

        records.append({"image": img_filename, "text": text})

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} images")

    csv_path = os.path.join(output_dir, "labels.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "text"])
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved {len(records)} records to {csv_path}")
    return csv_path


def split_dataset(csv_path: str, output_dir: str, train_ratio=0.7, val_ratio=0.15, seed=42):
    random.seed(seed)
    df = pd.read_csv(csv_path)

    writers = df["text"].apply(lambda x: hash(x) % 1000).unique().tolist()
    random.shuffle(writers)

    n = len(writers)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_writers = set(writers[:train_end])
    val_writers = set(writers[train_end:val_end])
    test_writers = set(writers[val_end:])

    train_df = df[df["text"].apply(lambda x: hash(x) % 1000).isin(train_writers)]
    val_df = df[df["text"].apply(lambda x: hash(x) % 1000).isin(val_writers)]
    test_df = df[df["text"].apply(lambda x: hash(x) % 1000).isin(test_writers)]

    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    def move_images(df, split_name):
        csv_rows = []
        for _, row in df.iterrows():
            src = os.path.join(output_dir, "images", row["image"])
            dst = os.path.join(output_dir, split_name, row["image"])
            if os.path.exists(src):
                os.rename(src, dst)
            csv_rows.append({"image": row["image"], "text": row["text"]})

        split_csv = os.path.join(output_dir, f"{split_name}.csv")
        with open(split_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "text"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  {split_name}: {len(csv_rows)} samples")

    print("\nSplitting dataset:")
    move_images(train_df, "train")
    move_images(val_df, "val")
    move_images(test_df, "test")

    stats = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "total": len(df),
    }
    with open(os.path.join(output_dir, "split_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSplit stats: {stats}")
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/bn_htrd")
    parser.add_argument("--skip_download", action="store_true")
    args = parser.parse_args()

    if not args.skip_download:
        csv_path = download_bn_htrd(args.output_dir)
    else:
        csv_path = os.path.join(args.output_dir, "labels.csv")

    split_dataset(csv_path, args.output_dir)
    print("\nData preparation complete!")
