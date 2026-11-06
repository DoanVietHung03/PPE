import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.core import YAMLConfig


CLASSES = ("person", "helmet", "vest", "gloves", "boots")
THRESHOLDS = (0.25, 0.4, 0.5, 0.6, 0.75)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyze D-FINE errors on full-frame test images")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[3] / "pipelines/ppe-5c/configs/dfine_m_960.yml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=960)
    return parser.parse_args(argv)


def load_full_frame_test_set(dataset):
    with (dataset / "manifest.csv").open(encoding="utf-8") as manifest_file:
        manifest_rows = list(csv.DictReader(manifest_file))
    manifest = {
        row["output_image"]: row
        for row in manifest_rows
        if row["split"] == "test" and row["image_kind"] == "full_frame"
    }
    coco = json.loads((dataset / "annotations/instances_test.json").read_text(encoding="utf-8"))
    names = {category["id"]: category["name"] for category in coco["categories"]}
    annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        x, y, width, height = annotation["bbox"]
        annotations[annotation["image_id"]].append(
            (names[annotation["category_id"]], (x, y, x + width, y + height), annotation["area"])
        )
    images = [image for image in coco["images"] if image["file_name"] in manifest]
    return images, annotations, names, manifest


def iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if not intersection:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def match(ground_truth, predictions, score_threshold, iou_threshold=0.5):
    predictions = sorted(
        (prediction for prediction in predictions if prediction[2] >= score_threshold),
        key=lambda prediction: prediction[2],
        reverse=True,
    )
    matched_targets, matched_predictions = set(), set()
    for prediction_index, prediction in enumerate(predictions):
        candidates = [
            (iou(target[1], prediction[1]), target_index)
            for target_index, target in enumerate(ground_truth)
            if target_index not in matched_targets and target[0] == prediction[0]
        ]
        if candidates:
            overlap, target_index = max(candidates)
            if overlap >= iou_threshold:
                matched_targets.add(target_index)
                matched_predictions.add(prediction_index)
    return matched_targets, matched_predictions, predictions


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(counts):
    true_positives, false_positives, false_negatives = counts["tp"], counts["fp"], counts["fn"]
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0
    return {
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
    }


def _load_model(args):
    config = YAMLConfig(str(args.config), resume=str(args.checkpoint))
    config.yaml_cfg["HGNetv2"]["pretrained"] = False
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = saved["ema"]["module"] if "ema" in saved else saved["model"]
    device = torch.device(args.device)
    model = config.model
    model.load_state_dict(state)
    return model.deploy().to(device).eval(), config.postprocessor.deploy().to(device).eval(), device


def analyze(args):
    images, annotations, _, manifest = load_full_frame_test_set(args.dataset)
    model, postprocessor, device = _load_model(args)
    predictions = {}
    started = time.time()
    for start in range(0, len(images), args.batch_size):
        batch = images[start : start + args.batch_size]
        tensors, sizes = [], []
        for image_info in batch:
            image = Image.open(args.dataset / image_info["file_name"]).convert("RGB")
            sizes.append((image.width, image.height))
            resized = image.resize((args.input_size, args.input_size))
            raw = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8)
            tensors.append(raw.view(args.input_size, args.input_size, 3).permute(2, 0, 1).float().div_(255))
        autocast = torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda")
        with torch.inference_mode(), autocast:
            labels, boxes, scores = postprocessor(
                model(torch.stack(tensors).to(device)), torch.tensor(sizes, device=device)
            )
        for image_info, labels_one, boxes_one, scores_one in zip(
            batch, labels.cpu(), boxes.cpu(), scores.cpu()
        ):
            predictions[image_info["id"]] = [
                (CLASSES[int(label)], tuple(map(float, box)), float(score))
                for label, box, score in zip(labels_one, boxes_one, scores_one)
            ]
        if start % 100 == 0:
            print(f"{min(start + args.batch_size, len(images))}/{len(images)}", flush=True)

    threshold_reports = {}
    for threshold in THRESHOLDS:
        counts = {class_name: Counter() for class_name in CLASSES}
        for image_info in images:
            ground_truth = [(name, box) for name, box, _ in annotations[image_info["id"]]]
            matched_gt, matched_pred, filtered = match(
                ground_truth, predictions[image_info["id"]], threshold
            )
            for index, (name, _) in enumerate(ground_truth):
                counts[name]["tp" if index in matched_gt else "fn"] += 1
            for index, (name, _, _) in enumerate(filtered):
                if index not in matched_pred:
                    counts[name]["fp"] += 1
        threshold_reports[str(threshold)] = {
            class_name: _metrics(counts[class_name]) for class_name in CLASSES
        }

    score_threshold = 0.5
    source_counts, size_counts = defaultdict(Counter), defaultdict(Counter)
    class_size_counts = {name: defaultdict(Counter) for name in CLASSES}
    ranked = []
    for image_info in images:
        annotated = annotations[image_info["id"]]
        ground_truth = [(name, box) for name, box, _ in annotated]
        matched_gt, matched_pred, filtered = match(
            ground_truth, predictions[image_info["id"]], score_threshold
        )
        false_negatives = [annotated[index] for index in range(len(ground_truth)) if index not in matched_gt]
        false_positives = [filtered[index] for index in range(len(filtered)) if index not in matched_pred]
        source = manifest[image_info["file_name"]]["source_id"]
        source_counts[source].update(tp=len(matched_gt), fn=len(false_negatives), fp=len(false_positives))
        for index, (name, _, area) in enumerate(annotated):
            bucket = "small" if area < 32**2 else "medium" if area < 96**2 else "large"
            outcome = "tp" if index in matched_gt else "fn"
            size_counts[bucket][outcome] += 1
            class_size_counts[name][bucket][outcome] += 1
        for name, box, _ in false_positives:
            area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
            bucket = "small" if area < 32**2 else "medium" if area < 96**2 else "large"
            size_counts[bucket]["fp"] += 1
            class_size_counts[name][bucket]["fp"] += 1
        ranked.append((len(false_negatives) * 2 + len(false_positives), image_info, false_negatives, false_positives))

    args.output.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output / "samples"
    samples_dir.mkdir(exist_ok=True)
    samples = []
    for rank, (error_score, image_info, false_negatives, false_positives) in enumerate(
        sorted(ranked, reverse=True, key=lambda item: item[0])[:50], 1
    ):
        image = Image.open(args.dataset / image_info["file_name"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for name, box, _ in false_negatives:
            draw.rectangle(box, outline="orange", width=3)
            draw.text((box[0], box[1]), f"FN {name}", fill="orange")
        for name, box, score in false_positives:
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], max(0, box[1] - 12)), f"FP {name} {score:.2f}", fill="red")
        filename = f"{rank:02d}_{image_info['id']}_{Path(image_info['file_name']).name}"
        image.save(samples_dir / filename, quality=90)
        samples.append({
            "file": filename,
            "source": manifest[image_info["file_name"]]["source_id"],
            "false_negatives": [item[0] for item in false_negatives],
            "false_positives": [{"class": item[0], "score": item[2]} for item in false_positives],
            "error_score": error_score,
        })

    report = {
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "benchmark_sha256": {
            "manifest.csv": _sha256(args.dataset / "manifest.csv"),
            "instances_test.json": _sha256(args.dataset / "annotations/instances_test.json"),
        },
        "images": len(images),
        "input_size": args.input_size,
        "iou": 0.5,
        "seconds": time.time() - started,
        "thresholds": threshold_reports,
        "by_source_at_0.5": {name: _metrics(counts) for name, counts in source_counts.items()},
        "by_size_at_0.5": {name: _metrics(counts) for name, counts in size_counts.items()},
        "by_class_and_size_at_0.5": {
            name: {bucket: _metrics(counts) for bucket, counts in buckets.items()}
            for name, buckets in class_size_counts.items()
        },
        "samples": samples,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    analyze(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
