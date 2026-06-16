KAGGLE NOTEBOOK CELL ORDER
===========================
Paste each file's content into a separate code cell in order.

Cell #  | File                       | What it does
--------|----------------------------|---------------------------------------------------------------
   1    | 01_clone_repo.py           | Clone the GitHub repo
   2    | 02_install_deps.py         | Install Python dependencies
   3    | 03_load_dataset.py         | Load BN-HTRd from Hugging Face
   4    | 04_save_images.py          | Save all images to disk + create labels.csv
   5    | 05_show_stats.py           | Display dataset statistics (total samples, unique words)
   6    | 06_split_dataset.py        | Split into train/val/test (70/15/15)
   7    | 07_train_model.py          | TRAIN the model with BnGraphemizer tokenizer (~2-5 hours)
   8    | 08_train_bpe_optional.py   | (Optional) Train model with BPE tokenizer for comparison
   9    | 09_evaluate.py             | Evaluate the trained model on test set (CER/WER)
  10    | 10_display_test_samples.py | Show 5 random test images with ground truth
  11    | 11_predict_image_function.py | Define predict_image() and test on first sample
  12    | 12_test_generalization.py  | Test on 5 random samples to check generalization
  13    | 13_upload_own_image.py     | Instructions for uploading your own image
  14    | 14_set_custom_image.py     | Set the path to your custom test image
  15    | 15_run_custom_prediction.py | Run model on your custom image and see result
  16    | 16_export_model.py         | Copy model to output directory for download

NOTE: Cells 3-6 must run sequentially in order (they share variables).
Cells 10-15 require Cell 7 (training) to have completed successfully.
