"""
# CELL 15: Run Prediction on Custom Image

Paste this into the 15th code cell.
"""
best_model = "/kaggle/working/checkpoints_bng/best_model.pt"
if os.path.exists(best_model) and CUSTOM_IMAGE_PATH and os.path.exists(CUSTOM_IMAGE_PATH):
    print("=" * 60)
    print("MODEL PREDICTION")
    print("=" * 60)
    pred = predict_image(best_model, CUSTOM_IMAGE_PATH, "bng")
    print(f"\nRecognized Text: {pred}")
    print("=" * 60)
else:
    print("Skipping: model or image not available yet.")
