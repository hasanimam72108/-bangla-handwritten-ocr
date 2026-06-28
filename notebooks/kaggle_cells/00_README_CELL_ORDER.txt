# KAGGLE NOTEBOOK CELL ORDER

Copy and paste the python files in this folder into your Kaggle Notebook cells in the following exact order:

## PRE-TRAINING PHASE
Cell 1: `01_clone_and_install.py` (Downloads code from GitHub and installs dependencies)
Cell 2: `02_prepare_data.py` (Aligns BN-HTRd dataset and creates train.csv & val.csv)
Cell 3: `03_inspect_crops.py` (Optional: visually verifies the dataset)
Cell 4: `04_train_model.py` (Starts the training loop)

## POST-TRAINING PHASE
*Wait for Cell 4 to completely finish training before running these!*
Cell 5: `05_evaluate_model.py` (Gets final metrics on the validation set)
Cell 6: `06_predict_sample.py` (Tests the model on a random validation image)
Cell 7: `07_custom_predict.py` (Tests the model on your own custom uploaded images)
Cell 8: `08_accuracy_graphs.py` (Generates line graphs of CER and WER history)
Cell 9: `09_error_matrix.py` (Generates the 2x2 error operations heatmap)
