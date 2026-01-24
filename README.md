# NeuroBase

**Transformer-based Foundation Model for EEG Seizure Detection**

A proof-of-concept foundation model for EEG signals, pretrained with masked patch prediction and fine-tuned for binary seizure classification on the CHB-MIT Scalp EEG Database.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quick test (verify everything works)
python main.py --test

# Launch interactive demo
python main.py --demo
```

## Full Pipeline

```bash
# Download CHB-MIT data (3 patients by default)
python main.py --download

# Pretrain encoder with masked patch prediction
python main.py --pretrain

# Fine-tune for seizure classification
python main.py --finetune

# Evaluate model
python main.py --eval

# Run everything
python main.py --all
```

## Architecture

| Component | Details |
|-----------|---------|
| Input | 18-channel EEG, 4-second windows (1024 samples @ 256 Hz) |
| Patch Embedding | 16 patches of 64 samples each |
| Transformer | 4 layers, 4 attention heads, 256 hidden dim |
| Parameters | ~2M |
| Pretraining | Masked patch prediction (40% mask ratio) |
| Fine-tuning | Binary classification on CLS token |

## Current Results

Trained on 3 patients from CHB-MIT (limited data):

| Metric | Score |
|--------|-------|
| Accuracy | 89.4% |
| Recall | 21.9% |
| Precision | 8.0% |
| F1 | 0.12 |
| AUROC | 0.50 |

**Note**: Performance is limited due to severe class imbalance (69 seizure vs 7000+ non-seizure windows) and small dataset. The model tends to predict "no seizure" as the safe default. This is a proof-of-concept demonstrating the architecture and pipeline.

## Project Structure

```
neurobase/
├── config.py          # All hyperparameters
├── main.py            # CLI entry point
├── data/              # Download, preprocessing, datasets
├── models/            # Encoder, pretraining, classifier, baselines
├── training/          # Training loops
├── evaluation/        # Metrics
├── demo/              # Streamlit app
├── scripts/           # CLI scripts
└── checkpoints/       # Saved models
```

## Dataset

[CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) from PhysioNet - pediatric subjects with intractable seizures.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- See `requirements.txt` for full list

## License

MIT
