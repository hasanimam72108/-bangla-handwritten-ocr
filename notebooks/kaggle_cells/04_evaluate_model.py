"""
# CELL 4: Evaluate the Model
Paste this into the fourth code cell.
This will evaluate the model on the validation set.
"""
!python evaluate.py --config configs/train_config.yaml --data_dir /kaggle/working/data --checkpoint /kaggle/working/checkpoints/best_model.pt
