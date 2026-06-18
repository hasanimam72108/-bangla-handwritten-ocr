KAGGLE NOTEBOOK CELL ORDER
===========================
Paste each file's content into a separate code cell in the following order.
This is the updated workflow perfectly tailored for your Bongabdo and BanglaWriting datasets.

Cell #  | File                       | What it does
--------|----------------------------|---------------------------------------------------------------
   1    | 01_clone_and_install.py    | Clone the GitHub repo and install requirements.txt
   2    | 02_prepare_data.py         | Runs the auto-cropping script to convert your full pages to lines
   3    | 03_train_model.py          | Trains the model on the newly cropped data
   4    | 04_evaluate_model.py       | Evaluates the trained model (CER/WER)
   5    | 05_predict_sample.py       | Picks a random validation image, shows it, and predicts text

NOTES:
- Ensure your Kaggle notebook has GPU T4 x2 enabled.
- Ensure you have added the two datasets via Kaggle's "Add Input" button.
- Wait for Cell 3 (Training) to completely finish before running Cells 4 and 5.
