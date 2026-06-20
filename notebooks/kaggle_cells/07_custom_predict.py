"""
# CELL 7: Predict on Custom Uploaded Image
Paste this into the seventh code cell.
First, upload your own handwritten Bengali image to Kaggle (e.g. name it 'my_test.jpg' and put it in /kaggle/working).
Then run this cell to see the model read it!
"""
import os
from IPython.display import display, Image as IPImage

# List your image paths here
test_images = [
    "/kaggle/input/datasets/hasanimam72180/test-1/dfghjuiuyghgfucgjhk.jpg",
    "/kaggle/input/datasets/hasanimam72180/test-1/kjhgfdsasdcfvgbnm.jpg",
    "/kaggle/input/datasets/hasanimam72180/test-1/erfbhtukdtmngzdfbsdga.jpg"
]

for img_path in test_images:
    print("\n" + "="*60)
    if os.path.exists(img_path):
        # Display the image visually in the notebook
        display(IPImage(filename=img_path))
        
        # Run the OCR prediction
        !python /kaggle/working/-bangla-handwritten-ocr/predict.py --image {img_path} --checkpoint /kaggle/working/checkpoints/best_model.pt --config /kaggle/working/-bangla-handwritten-ocr/configs/train_config.yaml --data_dir /kaggle/working/data
    else:
        print(f"Could not find image at: {img_path}")
print("="*60)
