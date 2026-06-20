"""
# CELL 5: Evaluate Model
Paste this into the fifth code cell.
Run this ONLY AFTER training is complete to evaluate your final model metrics.
"""
!python evaluate.py --config configs/train_config.yaml --data_dir /kaggle/working/data --checkpoint_path /kaggle/working/checkpoints/best_model.pt
