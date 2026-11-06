import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.ppe.registry import load_registry


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--path", type=Path, required=True)
    args = parser.parse_args(argv)
    model = load_registry(args.registry).require_approved_model(args.model)
    digest = hashlib.sha256(args.path.read_bytes()).hexdigest()
    if digest != model.sha256:
        raise ValueError(f"{args.model}: checksum mismatch")
    print(f"verified {args.model}: sha256={digest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
