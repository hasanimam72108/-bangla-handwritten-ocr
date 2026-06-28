# Chapter 4: Results and Discussion

## 4.1 Evaluation Metrics
The performance of the OCR pipeline was evaluated using three primary metrics commonly utilized in sequence recognition tasks:
1. **Character Error Rate (CER)**: The Levenshtein distance at the character level between the predicted string and the ground truth, normalized by the length of the ground truth. This is the most crucial metric for morphologically complex scripts like Bengali.
2. **Word Error Rate (WER)**: The Levenshtein distance at the word level (separated by spaces). It is heavily penalized in Bengali due to compounding conjuncts.
3. **Exact Match Accuracy**: The percentage of sequences where the entire predicted line perfectly matches the ground truth sequence without a single character deviation.

## 4.2 Quantitative Results
After initial training on the BN-HTRd dataset for 17 epochs using the base TrOCR architecture, the model achieved the following metrics on the isolated validation set (1,384 samples):
- **Character Accuracy**: 62.66% (CER: 0.3734)
- **Word Accuracy**: 50.70% (WER: 0.4930)
- **Exact Sequence Match**: 5.85%

While an Exact Match score of 5.85% appears statistically low, it is an expected outcome for line-level handwriting recognition where a single misplaced *matra* (diacritic) or punctuation mark invalidates the entire sentence. The Character Accuracy of 62.66% demonstrates that the model successfully learned the visual-to-text mapping of core Bengali graphemes from absolute scratch, successfully overriding its English-pretrained weights.

## 4.3 Qualitative Analysis and Model Behavior
A manual qualitative inspection of the model's predictions yielded deep insights into its functional behavior and limitations.

### 4.3.1 In-Distribution Success
When tested against the intrinsic vocabulary of the dataset (BBC News corpus), the model performed exceptionally well. It successfully reconstructed complex words and recognized formal journalistic vocabulary. The pipeline proved that the physical data alignment and custom tokenization strategies were fundamentally sound. For example, during custom inference, the model perfectly recognized the target word "নাম" (Name).

### 4.3.2 Out-of-Distribution (OOD) Language Bias
A significant limitation emerged during real-world custom testing. When the model was provided with casual, everyday handwritten questions (e.g., "তোমার মায়ের নাম কী ?"), it frequently hallucinated formal vocabulary. For instance, the word "তোমার" (Your) was visually misinterpreted and predicted as "আমার" (My), while "মায়ের" (Mother's) resulted in incoherent predictions like "বেকার" (Unemployed). 
This hallucination is directly attributed to the pre-trained nature of the decoder. Because it lacks a native understanding of general Bengali grammar, it acts as a probabilistic language model heavily biased toward the BBC News texts it observed during fine-tuning. When confronted with ambiguous cursive strokes, it forces a prediction of the most statistically likely word from its training distribution.

### 4.3.3 Sensitivity to Whitespace and Cropping
The model demonstrated extreme sensitivity to image aspect ratios and whitespace. The BN-HTRd dataset comprises images that are cropped tightly around the ink boundaries with near-zero margins. Consequently, when custom test images containing large amounts of vertical white space were passed to the `pad_to_square` function, the actual ink was scaled down significantly to fit the 384x384 tensor. This scale mismatch caused the Vision Transformer to misinterpret the strokes, leading to gibberish output.

## 4.4 Conclusion and Discussion
The implementation of the `microsoft/trocr-base-handwritten` model established a strong proof-of-concept for offline Bengali handwritten text recognition. The primary hurdles of data alignment, tokenizer corruption, and disk management were fully resolved, yielding a highly functional pipeline.

However, the empirical results conclude that adapting an English-first decoder to a complex script like Bengali imposes a hard ceiling on accuracy due to out-of-distribution vocabulary bias. Furthermore, the 224/384 ViT encoder limits the resolution needed to capture dense Bengali conjuncts. To breach the current 62.66% character accuracy, future iterations of this architecture must discard the English decoder. Transitioning to a hybrid model that pairs a high-resolution Vision Transformer (`vit-base-patch16-384`) with a decoder natively pre-trained on the Bengali language (such as `xlm-roberta-base`) is the definitive next step, coupled with heavy script-aware data augmentation (e.g., elastic deformations) to bridge the gap between dataset samples and real-world cursive handwriting.
