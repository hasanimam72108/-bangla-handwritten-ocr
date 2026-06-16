"""
# CELL 10: Display Test Samples

Paste this into the 10th code cell.
Shows ground truth images from the test set.
"""
from IPython.display import display
from PIL import Image as PILImage

test_csv = os.path.join(DATA_DIR, "test.csv")
if os.path.exists(test_csv):
    test_df = pd.read_csv(test_csv)
    print(f"Test set has {len(test_df)} images")
    
    samples = test_df.sample(min(5, len(test_df)))
    for _, row in samples.iterrows():
        img_path = os.path.join(DATA_DIR, "test", row["image"])
        if os.path.exists(img_path):
            img = PILImage.open(img_path)
            display(img)
            print(f"Ground truth: {row['text']}")
            print()
else:
    print("Test CSV not found. Run data preparation first.")
