import os
import sys
import yaml
import torch
import argparse
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.tokenizer import create_tokenizer
from src.models.ocr_model import build_model
from src.data.preprocessing import preprocess_pipeline, ImageTransform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--tokenizer", type=str, choices=["bpe", "bng"], default=None)
    parser.add_argument("--data_dir", type=str, default="data/bn_htrd")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, "r"))
    if args.tokenizer:
        config["tokenizer"]["type"] = args.tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizer...")
    texts = []
    import pandas as pd
    if os.path.exists(args.data_dir):
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
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}")

    print(f"\nLoading image: {args.image}")
    image = Image.open(args.image).convert("RGB")

    transform = ImageTransform(
        target_size=config["data"]["image_height"],
        augment=False,
    )
    pixel_values = transform(image).unsqueeze(0).to(device)

    print("Running inference...")
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_length=config["model"]["max_length"],
            num_beams=4,
            early_stopping=True,
        )

    prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("\n" + "=" * 50)
    print("Recognized Text:")
    print("=" * 50)
    print(prediction)
    print("=" * 50)

    return prediction


if __name__ == "__main__":
    main()
