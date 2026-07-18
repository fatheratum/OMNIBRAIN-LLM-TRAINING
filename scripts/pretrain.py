#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Make sure Python can find the local packages
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.vault_dataset import VaultDataset
from training.trainer import OmniBrainTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    print(f"Loading dataset from: {args.data_path}")
    dataset = VaultDataset(args.data_path)
    print(f"Loaded {len(dataset)} samples")

    print("Initializing trainer...")
    trainer = OmniBrainTrainer(output_dir=args.output_dir)

    print(f"Starting training for {args.epochs} epochs...")
    trainer.train(dataset, epochs=args.epochs, batch_size=args.batch_size)

    print(f"\nTraining complete. Checkpoints saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
