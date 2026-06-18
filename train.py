import os
import sys
import yaml
import torch
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import LineDataset, collate_fn
from src.data.tokenizer import create_tokenizer
from src.models.ocr_model import build_model, count_parameters
from src.training.trainer import Trainer


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--tokenizer", type=str, choices=["bpe", "bng"], default=None)
    parser.add_argument("--data_dir", type=str, default="data/bn_htrd")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    if args.tokenizer:
        config["tokenizer"]["type"] = args.tokenizer

    device = get_device()
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n--- Building tokenizer ---")
    texts = []
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(args.data_dir, f"{split}.csv")
        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            texts.extend(df["text"].tolist())

    tokenizer = create_tokenizer(
        config["tokenizer"]["type"],
        texts=texts,
        vocab_size=config["tokenizer"].get("vocab_size", 5000),
    )
    print(f"Tokenizer type: {config['tokenizer']['type']}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    print("\n--- Loading datasets ---")
    train_dataset = LineDataset(
        csv_file=os.path.join(args.data_dir, "train.csv"),
        image_dir=os.path.join(args.data_dir, "train"),
        tokenizer=tokenizer,
        target_size=config["data"]["image_height"],
        augment=config["data"]["augment"],
        max_length=config["model"]["max_length"],
    )
    val_dataset = LineDataset(
        csv_file=os.path.join(args.data_dir, "val.csv"),
        image_dir=os.path.join(args.data_dir, "val"),
        tokenizer=tokenizer,
        target_size=config["data"]["image_height"],
        augment=False,
        max_length=config["model"]["max_length"],
    )

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    print("\n--- Building model ---")
    model = build_model(
        encoder_name=config["model"]["encoder_name"],
        decoder_name=config["model"]["decoder_name"],
        tokenizer=tokenizer,
        max_length=config["model"]["max_length"],
        dropout=config["model"]["dropout"],
        freeze_encoder=True,
    )
    model.to(device)

    param_counts = count_parameters(model)
    print(f"Encoder params: {param_counts['encoder']:,}")
    print(f"Decoder params: {param_counts['decoder']:,}")
    print(f"Total params: {param_counts['total']:,}")
    print(f"Trainable params: {param_counts['trainable']:,}")

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    print("\n--- Starting training ---")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        config=config,
        device=device,
        output_dir=args.output_dir,
    )
    best_cer = trainer.run()

    print(f"\nTraining complete! Best validation CER: {best_cer:.4f}")
    print(f"Model saved to {os.path.join(args.output_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()
