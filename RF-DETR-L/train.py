#!/usr/bin/env python3
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune RF-DETR-L on PPE-13c")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "dataset/candidates/ppe-13c/rf-detr",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output/candidates/ppe-13c/rf-detr-large",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from rfdetr import RFDETRLarge
    except ImportError as error:
        raise SystemExit('Missing RF-DETR: pip install "rfdetr[train]"') from error

    model = RFDETRLarge()
    model.train(
        dataset_dir=str(args.dataset_dir),
        output_dir=str(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
    )


if __name__ == "__main__":
    main()
