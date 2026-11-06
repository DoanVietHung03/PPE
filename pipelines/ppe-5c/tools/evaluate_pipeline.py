import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DFINE_ROOT = ROOT / "D-FINE"
sys.path.insert(0, str(DFINE_ROOT))

from src.ppe.evaluation import coco_state_samples, evaluate_states


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--instances", type=Path)
    parser.add_argument("--states", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.fixture:
        data = json.loads(args.fixture.read_text(encoding="utf-8"))
        samples = data.get("samples", data)
        metadata = data.get("metadata", {})
    else:
        if not all((args.instances, args.states, args.predictions)):
            parser.error("provide --fixture or --instances, --states, and --predictions")
        instances = json.loads(args.instances.read_text(encoding="utf-8"))
        states = json.loads(args.states.read_text(encoding="utf-8"))
        predictions = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines()]
        samples = coco_state_samples(instances, states, predictions)
        metadata = {"instances": str(args.instances), "states": str(args.states)}
    report = evaluate_states(samples)
    report["metadata"] = metadata
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# PPE evaluation",
        "",
        "- Dataset build: {}".format(report["metadata"].get("dataset_build_id", "unspecified")),
        f"- People: {report['people']}",
        f"- Macro F1: {report['macro_f1']}",
        f"- No-PPE macro recall: {report['no_ppe_macro_recall']}",
        f"- False alarms/person: {report['false_alarms_per_person']}",
    ]
    (args.output / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
