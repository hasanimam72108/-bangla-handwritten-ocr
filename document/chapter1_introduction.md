# Chapter 1: Introduction

## 1.1 Background
Optical Character Recognition (OCR) for handwritten text remains a profoundly challenging problem in the domain of computer vision and natural language processing. While significant advancements have been achieved for Latin scripts due to the availability of extensive datasets and robust models, low-resource languages with complex morphological structures, such as Bengali, present unique hurdles. Bengali is the native language of over 230 million people globally, yet digitizing historical documents, handwritten notes, and administrative forms in Bengali is bottlenecked by the lack of highly accurate OCR systems. 

The Bengali script comprises 11 vowels, 39 consonants, and over 250 compound characters (conjuncts/যুক্তাক্ষর). Unlike English, Bengali characters are written beneath a continuous horizontal line known as the *matra*, and vowels are often represented as modifiers (diacritics) attached to the top, bottom, or sides of a base consonant. This two-dimensional spatial complexity, combined with the extreme inter-writer variability in cursive handwriting, renders traditional OCR approaches highly ineffective.

## 1.2 Problem Statement
Recent breakthroughs in deep learning, specifically the Transformer architecture, have revolutionized sequence-to-sequence tasks. The TrOCR (Transformer-based Optical Character Recognition) architecture demonstrated state-of-the-art results on English handwritten datasets by leveraging a Vision Transformer (ViT) encoder and a RoBERTa-based language model decoder. However, applying a model pre-trained primarily on English data to Bengali handwritten text exposes several critical challenges:
1. **Dataset Misalignment**: Ground truth mapping in large-scale datasets often suffers from physical sorting mismatches, leading to catastrophic 100% error rates if images and labels are not programmatically aligned using strict identifier mappings.
2. **Tokenizer Spacing Artefacts**: Standard pre-trained tokenizers inherently inject spaces between subword tokens during decoding, which breaks the continuous nature of Bengali graphemes and corrupts the textual output.
3. **Out-of-Distribution Vocabulary Bias**: Because the pre-trained decoder lacks native Bengali grammar intuition, it relies heavily on the distribution of its fine-tuning dataset, struggling to read real-world text that diverges from the training distribution.

## 1.3 Research Objectives
This study aims to address these challenges by building an end-to-end, robust pipeline for Bengali Handwritten Text Recognition. The primary objectives are:
1. To develop a flawless data-alignment pipeline for the BN-HTRd dataset by utilizing ground-truth Excel mappings to construct line-level text from word-level identifiers, ensuring absolute pixel-to-text accuracy.
2. To adapt the `microsoft/trocr-base-handwritten` model for Bengali by resolving tokenizer decoding artifacts and preserving native script structures.
3. To train, optimize, and evaluate the model on the BN-HTRd dataset within hardware-constrained environments (Kaggle), utilizing advanced training loops with mixed-precision, custom batching, and strict memory-saving checkpoint policies.
4. To analyze the performance of the model using standard metrics (Character Error Rate, Word Error Rate) and identify limitations concerning out-of-distribution real-world handwriting and whitespace variance.

## 1.4 Scope and Organization
The scope of this project is confined to offline handwritten text recognition at the line level. The pipeline assumes images are already segmented into individual lines of text. The remainder of this report is organized as follows: Chapter 2 reviews the relevant literature and existing methodologies for OCR. Chapter 3 details the system architecture, dataset preparation, preprocessing steps, and training configurations. Chapter 4 presents the quantitative and qualitative results, discussing the model's strengths and its limitations with real-world inputs. Finally, Chapter 5 outlines future improvements, including architectural transitions to multi-lingual decoders.
