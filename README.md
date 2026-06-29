# Bangla Handwritten Sentence Recognition OCR

A complete OCR system for recognizing handwritten Bangla sentences using a TrOCR-style architecture (ViT encoder + Transformer decoder).

## Architecture

- **Encoder**: ViT (Vision Transformer) — pretrained `google/vit-base-patch16-384`
- **Decoder**: GPT-2 style Transformer — trained from scratch
- **Tokenizer**: Two variants compared — BPE vs BnGraphemizer (grapheme-level tokenizer for Bengali)
- **Framework**: PyTorch + Hugging Face Transformers

## Dataset

**BN-HTRd** — 788 full-page handwritten Bangla document images from 150 writers, ~14,000 lines, 23,115 unique words. Source: BBC Bangla News.

## How to Run

### On Kaggle (Recommended)

1. Go to [Kaggle](https://kaggle.com) → Create New Notebook
2. Choose GPU accelerator (T4 x2 recommended)
3. Clone this repo in the first cell:
   ```
   !git clone https://github.com/hasanimam72108/-bangla-handwritten-ocr
   ```
4. Run all cells in `notebooks/kaggle_training.ipynb` (or paste the notebook content)
5. The notebook will:
   - Download BN-HTRd from Hugging Face
   - Extract line images
   - Train the model (~2-5 hours on T4)
   - Evaluate on test set
   - Show inference demo with uploaded images
6. Download `bangla_ocr_model.pt` from Kaggle Output

### Local Inference

```bash
pip install -r requirements.txt
python predict.py --image test.jpg --checkpoint bangla_ocr_model.pt --tokenizer bng
```

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── dataset.py          # Dataset and DataLoader
│   │   ├── preprocessing.py    # Image preprocessing
│   │   ├── augmentation.py     # Data augmentation
│   │   └── tokenizer.py        # BPE + BnGraphemizer
│   ├── models/
│   │   └── ocr_model.py        # ViT + Transformer decoder
│   ├── training/
│   │   ├── trainer.py          # Training loop
│   │   └── metrics.py          # CER, WER evaluation
│   └── inference/              # Inference utilities
├── configs/
│   ├── train_config.yaml
│   └── model_config.yaml
├── notebooks/
│   └── kaggle_training.ipynb   # Main Kaggle notebook
├── scripts/
│   └── prepare_data.py         # Dataset download + prep
├── train.py                    # Training entry point
├── evaluate.py                 # Evaluation entry point
├── predict.py                  # Inference entry point
└── requirements.txt
```

## Tokenizer Comparison

- **BPE**: Standard byte-pair encoding, ~5000 tokens
- **BnGraphemizer**: Grapheme-level tokenizer for Bengali, ~350 tokens

The grapheme tokenizer maps each visually distinct Bengali unit to one token, which improves recognition of compound characters and conjuncts.

## Citation

```bibtex
@misc{rahman2022bnhtrd,
  title={BN-HTRd: A Benchmark Dataset for Document Level Offline Bangla Handwritten Text Recognition (HTR) and Line Segmentation},
  author={Rahman, Md. Ataur and Tabassum, Nazifa and Paul, Mitu and Pal, Riya and Islam, Mohammad Khairul},
  year={2022},
  eprint={2206.08977},
  archivePrefix={arXiv},
}
```



## Authors

- **Hasan Imam** ([GitHub: hasanimam72108](https://github.com/hasanimam72108))
- **Jawadur Rafid** ([GitHub: jawadur13](https://github.com/jawadur13))