import os
import sys
import yaml
import torch
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import LineDataset, collate_fn
from src.data.tokenizer import create_tokenizer
from src.models.ocr_model import build_model
from src.training.metrics import evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/bn_htrd")
    parser.add_argument("--tokenizer", type=str, choices=["bpe", "bng"], default=None)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, "r"))
    if args.tokenizer:
        config["tokenizer"]["type"] = args.tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizer...")
    texts = []
    import pandas as pd
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(args.data_dir, f"{split}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            texts.extend(df["text"].tolist())
    tokenizer = create_tokenizer(
        config["tokenizer"]["type"],
        texts=texts,
        vocab_size=config["tokenizer"].get("vocab_size", 5000),
    )

    print(f"Loading {args.split} dataset...")
    dataset = LineDataset(
        csv_file=os.path.join(args.data_dir, f"{args.split}.csv"),
        image_dir=os.path.join(args.data_dir, args.split),
        tokenizer=tokenizer,
        target_size=config["data"]["image_height"],
        augment=False,
        max_length=config["model"]["max_length"],
    )

    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    print("Building model...")
    model = build_model(
        encoder_name=config["model"]["encoder_name"],
        decoder_name=config["model"]["decoder_name"],
        tokenizer=tokenizer,
        max_length=config["model"]["max_length"],
        dropout=config["model"]["dropout"],
        freeze_encoder=False,
    )
    model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {args.checkpoint}")

    print(f"\nEvaluating on {args.split} split ({len(dataset)} samples)...")
    metrics = evaluate_model(model, loader, tokenizer, device)

    print("\n" + "=" * 40)
    print(f"Test Results:")
    print(f"  CER (Character Error Rate):        {metrics['cer']:.4f}")
    print(f"  WER (Word Error Rate):             {metrics['wer']:.4f}")
    print(f"  Exact Match Accuracy:             {metrics['exact_match_accuracy']:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
