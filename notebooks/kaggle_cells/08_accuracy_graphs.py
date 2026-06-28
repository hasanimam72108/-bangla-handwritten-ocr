"""
# CELL 8: Draw Accuracy Graphs using Matplotlib
Paste this into the eighth code cell.
This script extracts training logs and generates CER and WER line graphs for your report.
"""
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dir = "/kaggle/working/checkpoints/logs"

if not os.path.exists(log_dir):
    print("❌ The logs folder is missing! Kaggle deletes /kaggle/working/ on restarts.")
    print("If you didn't download the logs folder before the session died, the history is gone.")
else:
    # Find the event file
    event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if "events" in f]
    if not event_files:
        print("❌ No event files found in the logs directory.")
    else:
        print("✅ Logs found! Drawing graphs...")
        # Load the event file
        event_acc = EventAccumulator(event_files[0])
        event_acc.Reload()

        tags = event_acc.Tags()['scalars']
        
        if 'CER/val' in tags and 'WER/val' in tags:
            cer_events = event_acc.Scalars('CER/val')
            wer_events = event_acc.Scalars('WER/val')
            
            epochs = [e.step for e in cer_events]
            cer_vals = [e.value for e in cer_events]
            wer_vals = [e.value for e in wer_events]
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, cer_vals, marker='o', linewidth=2, label='Character Error Rate (CER)')
            plt.plot(epochs, wer_vals, marker='s', linewidth=2, label='Word Error Rate (WER)', color='orange')
            
            plt.title('Validation Error Rates Over Epochs', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Error Rate (Lower is Better)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            
            # Make the chart look professional
            plt.tight_layout()
            plt.show()
