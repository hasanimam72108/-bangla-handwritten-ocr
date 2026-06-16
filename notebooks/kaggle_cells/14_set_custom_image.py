"""
# CELL 14: Set Custom Image Path

Paste this into the 14th code cell.
Replace the path with your own image location.
"""
# --- SETUP: Choose your input method ---

# Option A: Path to an image already on Kaggle or working dir
CUSTOM_IMAGE_PATH = None  # e.g., "/kaggle/input/my-test/image.jpg"

# Option B: Copy test image from BN-HTRd test set as an example:
if os.path.exists(test_csv):
    test_df = pd.read_csv(test_csv)
    sample = test_df.iloc[0]
    CUSTOM_IMAGE_PATH = os.path.join(DATA_DIR, "test", sample["image"])
    print(f"Using test image: {CUSTOM_IMAGE_PATH}")
    print(f"Ground truth: {sample['text']}")
