"""
# CELL 16: Save and Export Model

Paste this into the 16th (last) code cell.
After this, download bangla_ocr_model.pt from Kaggle Output.
"""
output_dir = "/kaggle/working"
best_model = os.path.join(output_dir, "checkpoints_bng", "best_model.pt")
if os.path.exists(best_model):
    !cp {best_model} {output_dir}/bangla_ocr_model.pt
    !ls -lh {output_dir}/bangla_ocr_model.pt
    print("\nModel saved to Kaggle working directory.")
    print("Download it from Kaggle Output tab after the run completes.")
else:
    print("Model not found.")
