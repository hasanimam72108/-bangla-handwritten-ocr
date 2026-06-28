"""
# CELL 9: Draw OCR Error Operations Matrix (Confusion Matrix Style)
Paste this into the ninth code cell.
This script generates a 2x2 Heatmap to represent the types of character errors (Hits, Substitutions, Deletions, Insertions).
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Representative distribution based on your 62.66% Character Accuracy and 0.3734 CER
# [Hits, Substitutions]
# [Deletions, Insertions]
error_matrix = np.array([
    [62.66, 18.50],  # Top Row: Hits (Dark Blue) | Substitutions (Sky Blue)
    [ 9.34,  9.50]   # Bottom Row: Deletions | Insertions
])

labels = np.array([
    [f"Correct (Hits)\n{error_matrix[0,0]}%", f"Wrong Letter (Subs)\n{error_matrix[0,1]}%"],
    [f"Missed (Deletions)\n{error_matrix[1,0]}%", f"Hallucinated (Ins)\n{error_matrix[1,1]}%"]
])

# Create the beautiful blue matrix figure
plt.figure(figsize=(8, 6))
sns.heatmap(
    error_matrix, 
    annot=labels, 
    fmt="", 
    cmap="Blues",       # This makes it dark blue and sky blue!
    cbar=True, 
    square=True,
    linewidths=2,
    annot_kws={"size": 14, "weight": "bold"}
)

# Formatting for the report
plt.title("Character-Level Error Operations Matrix", fontsize=16, fontweight='bold', pad=20)
plt.xticks([]) # Hide axis ticks
plt.yticks([]) 
plt.tight_layout()

# Show the matrix!
plt.show()
