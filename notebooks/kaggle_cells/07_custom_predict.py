"""
# CELL 7: Predict on Custom Uploaded Image
Paste this into the seventh code cell.
First, upload your own handwritten Bengali image to Kaggle (e.g. name it 'my_test.jpg' and put it in /kaggle/working).
Then run this cell to see the model read it!
"""
import os

# Change this to the exact path of your uploaded image!
my_image_path = '/kaggle/working/my_test.jpg'

if os.path.exists(my_image_path):
    !python /kaggle/working/-bangla-handwritten-ocr/predict.py --image {my_image_path} --checkpoint /kaggle/working/checkpoints/best_model.pt --config /kaggle/working/-bangla-handwritten-ocr/configs/train_config.yaml --data_dir /kaggle/working/data
else:
    print(f"I couldn't find the image at: {my_image_path}")
    print("Make sure you uploaded it and typed the path correctly!")
