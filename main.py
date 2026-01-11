#!/usr/bin/env python
"""
NeuroBase: Foundation Model for EEG Analysis

Main entry point for the complete pipeline:
1. Download data
2. Pretrain encoder
3. Fine-tune classifier
4. Evaluate
5. Launch demo

Usage:
    python main.py --download          # Download CHB-MIT data
    python main.py --pretrain          # Run pretraining
    python main.py --finetune          # Run fine-tuning
    python main.py --eval              # Run evaluation
    python main.py --demo              # Launch Streamlit demo
    python main.py --all               # Run complete pipeline
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from config import config


def download_data():
    """Download CHB-MIT dataset"""
    print("\n" + "="*60)
    print("STEP 1: DOWNLOADING DATA")
    print("="*60)
    
    from data import download_dataset
    
    all_patients = (
        config.data.train_patients + 
        config.data.val_patients + 
        config.data.test_patients
    )
    
    download_dataset(
        patients=all_patients,
        data_dir=config.data.data_dir,
        max_files_per_patient=8  # Limit for faster download
    )


def run_pretrain():
    """Run pretraining"""
    print("\n" + "="*60)
    print("STEP 2: PRETRAINING")
    print("="*60)
    
    subprocess.run([
        sys.executable, 
        "scripts/run_pretrain.py"
    ], check=True)


def run_finetune():
    """Run fine-tuning"""
    print("\n" + "="*60)
    print("STEP 3: FINE-TUNING")
    print("="*60)
    
    pretrained_path = config.training.checkpoint_dir / "best_pretrain.pt"
    
    cmd = [sys.executable, "scripts/run_finetune.py"]
    if pretrained_path.exists():
        cmd.extend(["--pretrained", str(pretrained_path)])
    
    subprocess.run(cmd, check=True)


def run_eval():
    """Run evaluation"""
    print("\n" + "="*60)
    print("STEP 4: EVALUATION")
    print("="*60)
    
    subprocess.run([
        sys.executable,
        "scripts/run_eval.py"
    ], check=True)


def run_demo():
    """Launch Streamlit demo"""
    print("\n" + "="*60)
    print("LAUNCHING DEMO")
    print("="*60)
    print("Starting Streamlit server...")
    print("Open http://localhost:8501 in your browser")
    print("Press Ctrl+C to stop")
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "demo/app.py",
        "--server.port", "8501"
    ])


def quick_test():
    """Quick test to verify everything works"""
    print("\n" + "="*60)
    print("QUICK TEST")
    print("="*60)
    
    import torch
    from models import EEGEncoder, SeizureClassifier, MaskedPatchPredictor
    
    print("Testing model creation...")
    encoder = EEGEncoder()
    print(f"  Encoder params: {encoder.get_num_params():,}")
    
    classifier = SeizureClassifier(encoder)
    print(f"  Classifier created")
    
    pretrainer = MaskedPatchPredictor(encoder)
    print(f"  Pretrainer created")
    
    print("\nTesting forward pass...")
    x = torch.randn(2, 18, 1024)
    
    # Test encoder
    out = encoder(x)
    print(f"  Encoder output: cls={out['cls_embedding'].shape}, patches={out['patch_embeddings'].shape}")
    
    # Test classifier
    out = classifier(x)
    print(f"  Classifier output: probs={out['probs'].shape}")
    
    # Test pretrainer
    out = pretrainer(x)
    print(f"  Pretrainer loss: {out['loss'].item():.4f}")
    
    print("\n✅ All tests passed!")


def main():
    parser = argparse.ArgumentParser(
        description="NeuroBase: Foundation Model for EEG Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --test              Quick test
    python main.py --download          Download data
    python main.py --pretrain          Run pretraining
    python main.py --finetune          Run fine-tuning  
    python main.py --eval              Evaluate models
    python main.py --demo              Launch demo app
    python main.py --all               Complete pipeline
        """
    )
    
    parser.add_argument('--test', action='store_true', help='Run quick test')
    parser.add_argument('--download', action='store_true', help='Download CHB-MIT data')
    parser.add_argument('--pretrain', action='store_true', help='Run pretraining')
    parser.add_argument('--finetune', action='store_true', help='Run fine-tuning')
    parser.add_argument('--eval', action='store_true', help='Run evaluation')
    parser.add_argument('--demo', action='store_true', help='Launch demo app')
    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    
    args = parser.parse_args()
    
    # If no args, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    print("="*60)
    print("🧠 NEUROBASE - EEG FOUNDATION MODEL")
    print("="*60)
    
    if args.test:
        quick_test()
    
    if args.all or args.download:
        download_data()
    
    if args.all or args.pretrain:
        run_pretrain()
    
    if args.all or args.finetune:
        run_finetune()
    
    if args.all or args.eval:
        run_eval()
    
    if args.demo:
        run_demo()
    
    if args.all:
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETE!")
        print("="*60)
        print("\nTo launch the demo:")
        print("  python main.py --demo")


if __name__ == "__main__":
    main()
