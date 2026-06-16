"""
# CELL 12: Test Generalization on 5 Random Test Samples

Paste this into the 12th code cell.
Tests whether the model can recognize unseen sentences.
"""
import random

best_model = "/kaggle/working/checkpoints_bng/best_model.pt"
if os.path.exists(best_model) and os.path.exists(test_csv):
    test_df = pd.read_csv(test_csv)
    
    print("Testing 5 random test samples:")
    print("=" * 60)
    
    sample_indices = random.sample(range(len(test_df)), min(5, len(test_df)))
    correct = 0
    
    for idx in sample_indices:
        sample = test_df.iloc[idx]
        img_path = os.path.join(DATA_DIR, "test", sample["image"])
        if os.path.exists(img_path):
            pred = predict_image(best_model, img_path, "bng")
            gt = sample["text"]
            match = pred.strip() == gt.strip()
            if match:
                correct += 1
            print(f"GT: {gt}")
            print(f"PR: {pred}")
            print(f"Match: {match}")
            print()
    
    print(f"Sample accuracy: {correct}/{len(sample_indices)}")
else:
    print("Model or test data not ready yet.")
