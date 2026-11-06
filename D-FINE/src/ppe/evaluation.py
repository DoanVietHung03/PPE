from collections import Counter


def _divide(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def _label_report(counts: Counter, total: int) -> dict:
    precision = _divide(counts["tp"], counts["tp"] + counts["fp"])
    recall = _divide(counts["tp"], counts["tp"] + counts["fn"])
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "evaluated": counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"],
        "true_positives": counts["tp"],
        "false_positives": counts["fp"],
        "false_negatives": counts["fn"],
        "true_negatives": counts["tn"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "unknown_coverage": counts["predicted_unknown"] / total if total else 0.0,
    }


def _xyxy(box):
    x, y, width, height = box
    return x, y, x + width, y + height


def _iou(coco_box, predicted_box):
    left, top, right, bottom = _xyxy(coco_box)
    intersection = max(0, min(right, predicted_box[2]) - max(left, predicted_box[0])) * max(
        0, min(bottom, predicted_box[3]) - max(top, predicted_box[1])
    )
    union = (right - left) * (bottom - top) + (
        predicted_box[2] - predicted_box[0]
    ) * (predicted_box[3] - predicted_box[1]) - intersection
    return intersection / union if union else 0.0


def coco_state_samples(instances: dict, states: dict, predictions: list[dict]) -> list[dict]:
    """Convert explicit COCO PPE states into per-person evaluation samples."""
    instance_names = {item["id"]: item["name"] for item in instances["categories"]}
    state_names = {item["id"]: item["name"] for item in states["categories"]}
    people_by_image = {}
    for annotation in instances["annotations"]:
        if instance_names[annotation["category_id"]] == "person":
            people_by_image.setdefault(annotation["image_id"], []).append(annotation["bbox"])

    evidence = {}
    for annotation in states["annotations"]:
        people = people_by_image.get(annotation["image_id"], [])
        x, y, width, height = annotation["bbox"]
        center = x + width / 2, y + height / 2
        containing = [
            (index, box)
            for index, box in enumerate(people)
            if box[0] <= center[0] <= box[0] + box[2]
            and box[1] <= center[1] <= box[1] + box[3]
        ]
        if not containing:
            continue
        person_index, _ = min(containing, key=lambda item: item[1][2] * item[1][3])
        label = state_names[annotation["category_id"]]
        base = label.removeprefix("no_")
        evidence.setdefault((annotation["image_id"], person_index, base), set()).add(
            label.startswith("no_")
        )

    predictions_by_image = {item["image_id"]: item.get("people", []) for item in predictions}
    samples = []
    for image_id, people in people_by_image.items():
        predicted_people = predictions_by_image.get(image_id, [])
        for person_index, person_box in enumerate(people):
            ground_truth = {}
            for (state_image_id, state_person_index, base), values in evidence.items():
                if (state_image_id, state_person_index) != (image_id, person_index):
                    continue
                if len(values) > 1:
                    ground_truth[base] = ground_truth[f"no_{base}"] = "unknown"
                elif True in values:
                    ground_truth[base], ground_truth[f"no_{base}"] = "absent", "present"
                else:
                    ground_truth[base], ground_truth[f"no_{base}"] = "present", "absent"
            if not ground_truth:
                continue
            matches = [
                (_iou(person_box, predicted["box"]), predicted)
                for predicted in predicted_people
            ]
            score, matched = max(matches, default=(0.0, None), key=lambda item: item[0])
            samples.append({
                "ground_truth": ground_truth,
                "prediction": matched.get("states", {}) if score >= 0.5 else {},
                "buckets": {},
            })
    return samples


def evaluate_states(samples: list[dict]) -> dict:
    labels = sorted(
        {
            label
            for sample in samples
            for side in ("ground_truth", "prediction")
            for label in sample.get(side, {})
        }
    )
    counters = {label: Counter() for label in labels}
    totals = Counter()
    buckets = Counter()
    for sample in samples:
        ground_truth = sample.get("ground_truth", {})
        prediction = sample.get("prediction", {})
        for key, value in sample.get("buckets", {}).items():
            buckets[f"{key}={value}"] += 1
        for label in labels:
            if label not in ground_truth:
                continue
            totals[label] += 1
            expected = ground_truth[label]
            actual = prediction.get(label, "unknown")
            if actual == "unknown":
                counters[label]["predicted_unknown"] += 1
            if expected == "unknown":
                continue
            if expected == "present":
                counters[label]["tp" if actual == "present" else "fn"] += 1
            elif actual != "unknown":
                counters[label]["fp" if actual == "present" else "tn"] += 1
    label_reports = {
        label: _label_report(counters[label], totals[label]) for label in labels
    }
    f1_values = [report["f1"] for report in label_reports.values() if report["f1"] is not None]
    no_ppe_recalls = [
        report["recall"]
        for label, report in label_reports.items()
        if label.startswith("no_") and report["recall"] is not None
    ]
    false_positives = sum(report["false_positives"] for report in label_reports.values())
    bucket_reports = {}
    for name in sorted(buckets):
        key, value = name.split("=", 1)
        bucket_reports[name] = evaluate_states([
            {**sample, "buckets": {}}
            for sample in samples
            if str(sample.get("buckets", {}).get(key)) == value
        ])
    return {
        "people": len(samples),
        "labels": label_reports,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "no_ppe_macro_recall": (
            sum(no_ppe_recalls) / len(no_ppe_recalls) if no_ppe_recalls else None
        ),
        "false_alarms_per_person": false_positives / len(samples) if samples else 0.0,
        "buckets": bucket_reports,
    }
