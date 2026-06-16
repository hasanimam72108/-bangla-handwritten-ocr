"""
# CELL 5: Show Dataset Statistics

Paste this into the fifth code cell.
"""
df = pd.read_csv(csv_path)
print(f"Total samples: {len(df)}")
print(f"Unique words: {len(set(' '.join(df['text'].tolist()).split()))}")
print(f"\nSample rows:")
df.head(5)
