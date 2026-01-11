# NeuroBase

**Foundation Model for EEG Analysis - Seizure Detection Demo**

A transformer-based foundation model for EEG signals, pretrained with masked patch prediction and fine-tuned for seizure detection on the CHB-MIT dataset.

## Quick Start

```bash
# Setup
cd neurobase
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Quick test
python main.py --test

# Download data (5 patients)
python main.py --download

# Run full pipeline
python main.py --all

# Launch demo
python main.py --demo
```

## Project Structure

- `config.py` - Configuration
- `main.py` - Main entry point  
- `data/` - Data pipeline (download, preprocessing, datasets)
- `models/` - Model architectures (encoder, pretrain, classifier, baselines)
- `training/` - Training loops
- `evaluation/` - Metrics and comparison
- `demo/` - Streamlit app
- `scripts/` - CLI scripts

## Architecture

- Patch Embedding: 18 channels x 1024 samples -> 16 patches x 256 dim
- Transformer: 4 layers, 4 heads, 256 hidden dim
- Total: ~2M parameters
- Pretraining: Masked patch prediction (40% mask ratio)
- Fine-tuning: Binary classification on CLS token

## Dataset

CHB-MIT Scalp EEG Database from PhysioNet

## License

MIT
