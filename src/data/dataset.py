import os
import pandas as pd
import torch
from typing import Optional, Tuple, List
from torch.utils.data import Dataset
from PIL import Image

from src.data.preprocessing import ImageTransform


class LineDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        image_dir: str,
        tokenizer,
        target_size: int = 384,
        augment: bool = False,
        max_length: int = 256,
    ):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = ImageTransform(target_size=target_size, augment=augment)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image"])
        text = str(row["text"])

        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)

        tokens = self.tokenizer._tokenize(text)
        token_ids = [self.tokenizer._convert_token_to_id(t) for t in tokens]
        token_ids = token_ids[:self.max_length - 1] + [self.tokenizer.eos_token_id]
        pad_len = self.max_length - len(token_ids)
        if pad_len > 0:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * pad_len

        labels = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = (labels != self.tokenizer.pad_token_id).long()

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "attention_mask": attention_mask,
            "text": text,
        }


def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    texts = [item["text"] for item in batch]
    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "attention_mask": attention_mask,
        "texts": texts,
    }
