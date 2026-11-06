import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


DETECTOR_CLASSES = ("person", "helmet", "vest", "gloves", "boots")
PUBLIC_STATES = ("helmet", "vest", "gloves", "boots", "no_helmet", "no_vest", "no_gloves", "no_boots")


@dataclass(frozen=True)
class BuildReport:
    build_id: str
    path: Path
    audit: dict


class _Groups:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def _dhash(path: Path) -> str:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((9, 8)).get_flattened_data())
    bits = [pixels[row * 9 + column] > pixels[row * 9 + column + 1] for row in range(8) for column in range(8)]
    value = sum(bit << index for index, bit in enumerate(bits))
    return f"{value:016x}"


def _deduplicate_records(records, max_hamming=4):
    # ponytail: quadratic scan is simplest for current dataset size; add a BK-tree if builds become slow.
    kept = []
    fingerprints = []
    for record in records:
        fingerprint = int(_dhash(record.image_path), 16)
        if any(
            fingerprint == previous
            or (
                record.width == other.width
                and record.height == other.height
                and (fingerprint ^ previous).bit_count() <= max_hamming
            )
            for other, previous in zip(kept, fingerprints)
        ):
            continue
        kept.append(record)
        fingerprints.append(fingerprint)
    return tuple(kept), len(records) - len(kept)


def _group_records(records):
    # ponytail: exact dHash catches identical transforms; use Hamming-distance indexing if near-duplicate leakage appears.
    groups = _Groups(len(records))
    seen = {}
    hashes = []
    for index, record in enumerate(records):
        sha256 = hashlib.sha256(record.image_path.read_bytes()).hexdigest()
        perceptual = _dhash(record.image_path)
        hashes.append((sha256, perceptual))
        for key in (("origin", record.origin_group), ("sha256", sha256), ("dhash", perceptual)):
            if key in seen:
                groups.union(index, seen[key])
            else:
                seen[key] = index
    grouped = {}
    for index in range(len(records)):
        grouped.setdefault(groups.find(index), []).append(index)
    return grouped, hashes


def _assign_splits(records, groups, seed):
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    group_features = {}
    for group, indices in groups.items():
        features = Counter()
        for index in indices:
            record = records[index]
            features["images"] += 1
            features[f"source:{record.source_id}"] += 1
            for category in {annotation.category for annotation in record.annotations}:
                features[f"class:{category}"] += 1
        group_features[group] = features
    totals = sum(group_features.values(), Counter())
    counts = {split: Counter() for split in ratios}
    assignments = {}
    ordered = sorted(
        groups,
        key=lambda group: (
            -sum(group_features[group][feature] / totals[feature] for feature in group_features[group]),
            hashlib.sha256(
                f"{seed}:".encode()
                + ",".join(sorted(records[index].id for index in groups[group])).encode()
            ).digest(),
        ),
    )
    for group in ordered:
        def error(candidate):
            return sum(
                (
                    counts[split][feature]
                    + (group_features[group][feature] if split == candidate else 0)
                    - total * ratio
                ) ** 2
                / max(total * ratio, 1)
                for split, ratio in ratios.items()
                for feature, total in totals.items()
            )

        split = min(ratios, key=lambda name: (error(name), name != "train", name))
        for index in groups[group]:
            assignments[index] = split
        counts[split].update(group_features[group])
    return assignments


def _crop_box(bbox, image_width, image_height):
    x, y, width, height = bbox
    pad_x, pad_y = width * 0.1, height * 0.1
    left = max(0, int(x - pad_x))
    top = max(0, int(y - pad_y))
    right = min(image_width, int(x + width + pad_x + 0.999))
    bottom = min(image_height, int(y + height + pad_y + 0.999))
    return left, top, right, bottom


def _context_crop_box(bbox, image_width, image_height, scale=4.0):
    x, y, width, height = bbox
    crop_width = min(image_width, max(width * scale, 64))
    crop_height = min(image_height, max(height * scale, 64))
    center_x, center_y = x + width / 2, y + height / 2
    left = max(0, min(image_width - crop_width, center_x - crop_width / 2))
    top = max(0, min(image_height - crop_height, center_y - crop_height / 2))
    return int(left), int(top), int(left + crop_width + 0.999), int(top + crop_height + 0.999)


def _box_in_crop(annotation, bounds):
    left, top, right, bottom = bounds
    x, y, width, height = annotation.bbox
    center_x, center_y = x + width / 2, y + height / 2
    if not (left <= center_x <= right and top <= center_y <= bottom):
        return None
    crop_right, crop_bottom = min(x + width, right) - left, min(y + height, bottom) - top
    return max(x, left) - left, max(y, top) - top, crop_right - max(x, left) + left, crop_bottom - max(y, top) + top


def _safe_name(record_id: str, suffix: str) -> str:
    stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in record_id)
    return f"{stem}{suffix}"


def _canonical_build_id(records, hashes, seed, rejections, licenses, taxonomy):
    digest = hashlib.sha256(f"ppe-builder-v2:{seed}\n".encode())
    for record, (sha256, _) in sorted(zip(records, hashes), key=lambda item: item[0].id):
        digest.update(f"{record.id}:{record.origin_group}:{sha256}:{record.annotations!r}\n".encode())
    digest.update(json.dumps({"rejections": rejections, "licenses": licenses, "taxonomy": taxonomy}, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:16]


def _write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(
    records, output_dir: Path, seed: int = 20260901, *, rejections=(), licenses=None,
    detector_classes=DETECTOR_CLASSES, state_classes=PUBLIC_STATES,
    context_classes=("gloves", "boots"),
) -> BuildReport:
    detector_classes, state_classes, context_classes = map(tuple, (detector_classes, state_classes, context_classes))
    if not detector_classes or any(len(items) != len(set(items)) for items in (detector_classes, state_classes, context_classes)):
        raise ValueError("taxonomies must be non-empty and contain unique names")
    records = tuple(records)
    rejections = tuple(sorted(rejections, key=lambda row: tuple(str(row.get(key, "")) for key in ("source_id", "source_image_id", "code", "detail"))))
    require_licenses = licenses is not None
    licenses = tuple(sorted(licenses or (), key=lambda row: str(row.get("source_id", ""))))
    if not records:
        raise ValueError("Cannot build an empty dataset")
    records, deduplicated_full_frames = _deduplicate_records(records)
    records = tuple(sorted(records, key=lambda record: record.id))
    if require_licenses:
        licensed_sources = {row.get("source_id") for row in licenses if row.get("license") and row.get("evidence_url")}
        missing = sorted({record.source_id for record in records} - licensed_sources)
        if missing:
            raise ValueError(f"missing license evidence for source: {missing[0]}")
    groups, hashes = _group_records(records)
    assignments = _assign_splits(records, groups, seed)
    build_id = _canonical_build_id(records, hashes, seed, rejections, licenses, {"detector": detector_classes, "states": state_classes, "context": context_classes})
    output_dir = Path(output_dir)
    final = output_dir / build_id
    if final.exists():
        audit = json.loads((final / "audit.json").read_text(encoding="utf-8"))
        return BuildReport(build_id, final, audit)

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}-", dir=output_dir))
    manifest = []
    coco = {
        split: {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": index, "name": name} for index, name in enumerate(detector_classes)
            ],
        }
        for split in ("train", "val", "test")
    }
    states = {
        split: {
            "images": [],
            "annotations": [],
            "categories": [{"id": index, "name": name} for index, name in enumerate(state_classes)],
        }
        for split in ("train", "val", "test")
    }
    next_image_id = 1
    next_annotation_id = 1
    next_state_annotation_id = 1
    try:
        for index, record in enumerate(records):
            split = assignments[index]
            image_dir = temporary / "images" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            suffix = record.image_path.suffix.lower() or ".jpg"
            image_name = _safe_name(record.id, suffix)
            shutil.copy2(record.image_path, image_dir / image_name)
            manifest.append(
                {
                    "record_id": record.id,
                    "parent_id": "",
                    "source_id": record.source_id,
                    "source_image_id": record.source_image_id,
                    "origin_group": record.origin_group,
                    "sha256": hashes[index][0],
                    "dhash": hashes[index][1],
                    "split": split,
                    "image_kind": "full_frame",
                    "output_image": f"images/{split}/{image_name}",
                }
            )
            coco[split]["images"].append(
                {"id": next_image_id, "file_name": f"images/{split}/{image_name}", "width": record.width, "height": record.height}
            )
            states[split]["images"].append(dict(coco[split]["images"][-1]))
            for annotation in record.annotations:
                if annotation.category in detector_classes:
                    coco[split]["annotations"].append(
                        {
                            "id": next_annotation_id,
                            "image_id": next_image_id,
                            "category_id": detector_classes.index(annotation.category),
                            "bbox": list(annotation.bbox),
                            "area": annotation.bbox[2] * annotation.bbox[3],
                            "iscrowd": 0,
                        }
                    )
                    next_annotation_id += 1
            for annotation in record.annotations:
                if annotation.category in state_classes:
                    states[split]["annotations"].append({"id": next_state_annotation_id, "image_id": next_image_id, "category_id": state_classes.index(annotation.category), "bbox": list(annotation.bbox), "area": annotation.bbox[2] * annotation.bbox[3], "iscrowd": 0, "source_label": annotation.source_label})
                    next_state_annotation_id += 1
            next_image_id += 1

            people = [annotation for annotation in record.annotations if annotation.category == "person"]
            with Image.open(record.image_path) as source_image:
                for person_index, person in enumerate(people):
                    left, top, right, bottom = _crop_box(person.bbox, record.width, record.height)
                    crop_id = f"{record.id}#person-{person_index}"
                    crop_name = _safe_name(crop_id, ".jpg")
                    source_image.crop((left, top, right, bottom)).convert("RGB").save(image_dir / crop_name, quality=95)
                    manifest.append(
                        {
                            "record_id": crop_id,
                            "parent_id": record.id,
                            "source_id": record.source_id,
                            "source_image_id": record.source_image_id,
                            "origin_group": record.origin_group,
                            "sha256": hashlib.sha256((image_dir / crop_name).read_bytes()).hexdigest(),
                            "dhash": _dhash(image_dir / crop_name),
                            "split": split,
                            "image_kind": "person_crop",
                            "output_image": f"images/{split}/{crop_name}",
                        }
                    )
                    crop_width, crop_height = right - left, bottom - top
                    coco[split]["images"].append(
                        {
                            "id": next_image_id,
                            "file_name": f"images/{split}/{crop_name}",
                            "width": crop_width,
                            "height": crop_height,
                        }
                    )
                    states[split]["images"].append(dict(coco[split]["images"][-1]))
                    for annotation in record.annotations:
                        if annotation.category not in detector_classes:
                            continue
                        x, y, width, height = annotation.bbox
                        center_x, center_y = x + width / 2, y + height / 2
                        if not (left <= center_x <= right and top <= center_y <= bottom):
                            continue
                        crop_x = max(x, left) - left
                        crop_y = max(y, top) - top
                        crop_right = min(x + width, right) - left
                        crop_bottom = min(y + height, bottom) - top
                        crop_box = (crop_x, crop_y, crop_right - crop_x, crop_bottom - crop_y)
                        coco[split]["annotations"].append(
                            {
                                "id": next_annotation_id,
                                "image_id": next_image_id,
                                "category_id": detector_classes.index(annotation.category),
                                "bbox": list(crop_box),
                                "area": crop_box[2] * crop_box[3],
                                "iscrowd": 0,
                            }
                        )
                        next_annotation_id += 1
                    for annotation in record.annotations:
                        if annotation.category not in state_classes:
                            continue
                        x, y, width, height = annotation.bbox
                        center_x, center_y = x + width / 2, y + height / 2
                        if not (left <= center_x <= right and top <= center_y <= bottom):
                            continue
                        crop_x = max(x, left) - left
                        crop_y = max(y, top) - top
                        crop_right = min(x + width, right) - left
                        crop_bottom = min(y + height, bottom) - top
                        crop_box = (crop_x, crop_y, crop_right - crop_x, crop_bottom - crop_y)
                        states[split]["annotations"].append({"id": next_state_annotation_id, "image_id": next_image_id, "category_id": state_classes.index(annotation.category), "bbox": list(crop_box), "area": crop_box[2] * crop_box[3], "iscrowd": 0, "source_label": annotation.source_label})
                        next_state_annotation_id += 1
                    next_image_id += 1

            rare = [annotation for annotation in record.annotations if annotation.category in context_classes]
            with Image.open(record.image_path) as source_image:
                for rare_index, target in enumerate(rare):
                    bounds = _context_crop_box(target.bbox, record.width, record.height)
                    left, top, right, bottom = bounds
                    crop_id = f"{record.id}#{target.category}-context-{rare_index}"
                    crop_name = _safe_name(crop_id, ".jpg")
                    crop_path = image_dir / crop_name
                    source_image.crop(bounds).convert("RGB").save(crop_path, quality=95)
                    manifest.append({"record_id": crop_id, "parent_id": record.id, "source_id": record.source_id, "source_image_id": record.source_image_id, "origin_group": record.origin_group, "sha256": hashlib.sha256(crop_path.read_bytes()).hexdigest(), "dhash": _dhash(crop_path), "split": split, "image_kind": f"{target.category}_context", "output_image": f"images/{split}/{crop_name}"})
                    crop_image = {"id": next_image_id, "file_name": f"images/{split}/{crop_name}", "width": right - left, "height": bottom - top}
                    coco[split]["images"].append(crop_image)
                    states[split]["images"].append(dict(crop_image))
                    for annotation in record.annotations:
                        crop_box = _box_in_crop(annotation, bounds)
                        if crop_box is None:
                            continue
                        if annotation.category in detector_classes:
                            coco[split]["annotations"].append({"id": next_annotation_id, "image_id": next_image_id, "category_id": detector_classes.index(annotation.category), "bbox": list(crop_box), "area": crop_box[2] * crop_box[3], "iscrowd": 0})
                            next_annotation_id += 1
                        if annotation.category in state_classes:
                            states[split]["annotations"].append({"id": next_state_annotation_id, "image_id": next_image_id, "category_id": state_classes.index(annotation.category), "bbox": list(crop_box), "area": crop_box[2] * crop_box[3], "iscrowd": 0, "source_label": annotation.source_label})
                            next_state_annotation_id += 1
                    next_image_id += 1

        annotation_dir = temporary / "annotations"
        annotation_dir.mkdir()
        for split, data in coco.items():
            (annotation_dir / f"instances_{split}.json").write_text(
                json.dumps(data, sort_keys=True, separators=(",", ":")), encoding="utf-8"
            )
        for split, data in states.items():
            (annotation_dir / f"states_{split}.json").write_text(
                json.dumps(data, sort_keys=True, separators=(",", ":")), encoding="utf-8"
            )
        _write_csv(
            temporary / "manifest.csv",
            manifest,
            ["record_id", "parent_id", "source_id", "source_image_id", "origin_group", "sha256", "dhash", "split", "image_kind", "output_image"],
        )
        _write_csv(temporary / "rejected.csv", rejections, ["source_id", "source_image_id", "code", "detail"])
        _write_csv(temporary / "licenses.csv", licenses, ["source_id", "license", "evidence_url", "version", "sha256"])
        attribution = ["# Dataset attribution", ""]
        attribution.extend("- {}: {} — {}".format(row["source_id"], row["license"], row["evidence_url"]) for row in licenses)
        (temporary / "ATTRIBUTION.md").write_text("\n".join(attribution) + "\n", encoding="utf-8")
        box_sizes = Counter()
        sha_counts = Counter(item[0] for item in hashes)
        dhash_counts = Counter(item[1] for item in hashes)
        for record in records:
            for annotation in record.annotations:
                if annotation.category not in detector_classes:
                    continue
                ratio = annotation.bbox[2] * annotation.bbox[3] / (record.width * record.height)
                box_sizes["small" if ratio < 0.02 else "medium" if ratio < 0.15 else "large"] += 1
        audit = {
            "build_id": build_id,
            "images": len(manifest),
            "full_frames": len(records),
            "deduplicated_full_frames": deduplicated_full_frames,
            "person_crops": sum(row["image_kind"] == "person_crop" for row in manifest),
            "context_crops": sum(row["image_kind"].endswith("_context") for row in manifest),
            "origin_groups": len(groups),
            "cross_split_duplicate_groups": sum(len({assignments[index] for index in indices}) > 1 for indices in groups.values()),
            "classes": dict(sorted(Counter(annotation.category for record in records for annotation in record.annotations if annotation.category in detector_classes).items())),
            "sources": dict(sorted(Counter(record.source_id for record in records).items())),
            "licenses": dict(sorted(Counter(row["license"] for row in licenses).items())),
            "rejected_reasons": dict(sorted(Counter(row["code"] for row in rejections).items())),
            "box_size_buckets": dict(sorted(box_sizes.items())),
            "exact_duplicate_groups": sum(count > 1 for count in sha_counts.values()),
            "perceptual_duplicate_groups": sum(count > 1 for count in dhash_counts.values()),
            "splits": {split: sum(row["split"] == split for row in manifest) for split in ("train", "val", "test")},
        }
        (temporary / "audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, final)
        return BuildReport(build_id, final, audit)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
