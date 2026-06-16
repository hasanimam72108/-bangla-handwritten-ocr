"""
# CELL 7: Train Model (BnGraphemizer tokenizer — recommended)

Paste this into the seventh code cell.
Training takes ~2-5 hours on Kaggle T4 GPU.
"""
!python train.py --tokenizer bng --data_dir /kaggle/working/data/bangla_combined --output_dir /kaggle/working/checkpoints_bng
