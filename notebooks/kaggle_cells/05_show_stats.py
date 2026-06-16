"""
# CELL 5: Show Dataset Statistics

Paste this into the fifth code cell.
"""
df = pd.read_csv(csv_path)
all_text = " ".join(df["text"].tolist())
all_words = all_text.split()
unique_words = set(all_words)

print(f"Total samples: {len(df)}")
print(f"Total words: {len(all_words)}")
print(f"Unique words: {len(unique_words)}")
print(f"Total characters: {len(all_text)}")
print(f"Min text length: {df['text'].str.len().min()}")
print(f"Max text length: {df['text'].str.len().max()}")
print(f"Avg text length: {df['text'].str.len().mean():.1f}")
print(f"\nSample rows:")
df.head(5)
