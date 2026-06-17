"""
# CELL 11: Define Prediction Function and Test on First Sample

Paste this into the 11th code cell.
This defines the predict_image() function and tests it on the first test sample.
"""
import torch
import sys
import yaml

def predict_image(model_path, image_path, tokenizer_type="bng"):
    from src.data.tokenizer import create_tokenizer
    from src.models.ocr_model import build_model
    from src.data.preprocessing import ImageTransform
    
    config_path = os.path.join(REPO_DIR, "configs", "train_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["tokenizer"]["type"] = tokenizer_type
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    texts = []
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(DATA_DIR, f"{split}.csv")
        if os.path.exists(csv_path):
            df_local = pd.read_csv(csv_path)
            texts.extend(df_local["text"].tolist())
    tokenizer = create_tokenizer(tokenizer_type, texts=texts)
    
    # Load model
    model = build_model(
        encoder_name=config["model"]["encoder_name"],
        decoder_name=config["model"]["decoder_name"],
        tokenizer=tokenizer,
        max_length=config["model"]["max_length"],
        dropout=config["model"]["dropout"],
        freeze_encoder=True,
    )
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Preprocess image
    img = PILImage.open(image_path).convert("RGB")
    display(img)
    
    transform = ImageTransform(target_size=config["data"]["image_height"], augment=False)
    pixel_values = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_length=config["model"]["max_length"],
            num_beams=4,
            early_stopping=True,
        )
    
    prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return prediction


# Test the model on a sample
best_model = "/kaggle/working/checkpoints_bng/best_model.pt"
if os.path.exists(best_model):
    test_csv = os.path.join(DATA_DIR, "test.csv")
    if os.path.exists(test_csv):
        test_df = pd.read_csv(test_csv)
        sample = test_df.iloc[0]
        img_path = sample["image"] if os.path.isabs(sample["image"]) else os.path.join(DATA_DIR, "test", sample["image"])
        if os.path.exists(img_path):
            print(f"Ground truth: {sample['text']}")
            pred = predict_image(best_model, img_path, "bng")
            print(f"Prediction:   {pred}")
            print(f"Match: {pred.strip() == sample['text'].strip()}")
else:
    print("No trained model found. Run the training cell first.")
