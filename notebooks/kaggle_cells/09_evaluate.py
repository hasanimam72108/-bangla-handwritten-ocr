"""
# CELL 9: Evaluate Model on Test Set

Paste this into the ninth code cell.
"""
best_model = "/kaggle/working/checkpoints_bng/best_model.pt"
if os.path.exists(best_model):
    !python evaluate.py --checkpoint {best_model} --tokenizer bng --data_dir /kaggle/working/data/bn_htrd
else:
    print("Model not found yet. Run training first.")
