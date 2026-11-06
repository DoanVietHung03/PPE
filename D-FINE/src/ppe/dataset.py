import hashlib
import json
import os
import stat
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping

import yaml
from PIL import Image

from .registry import SourceSpec


class HashReviewRequired(RuntimeError):
    def __init__(self, source_id: str, sha256: str, path: Path):
        super().__init__(f"{source_id}: add sha256 {sha256} to the registry and review again")
        self.source_id = source_id
        self.sha256 = sha256
        self.path = path


def safe_extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe archive path: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"archive symlink is not allowed: {member.filename}")
        destination.mkdir(parents=True, exist_ok=True)
        bundle.extractall(destination)


def _roboflow_url(
    source: SourceSpec,
    env: Mapping[str, str],
    opener: Callable,
) -> tuple[str, dict[str, str]]:
    api_key = env.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY is required for Roboflow sources")
    workspace = source.options.get("workspace")
    project = source.options.get("project")
    if not workspace or not project:
        raise ValueError(f"{source.id}: missing Roboflow workspace or project")
    query = urllib.parse.urlencode({"api_key": api_key})
    url = f"https://api.roboflow.com/{workspace}/{project}/{source.version}/coco?{query}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with opener(request) as response:
        metadata = json.load(response)
    try:
        return metadata["export"]["link"], {}
    except (KeyError, TypeError) as error:
        raise ValueError(f"{source.id}: Roboflow response has no export link") from error


def _mendeley_url(
    source: SourceSpec,
    env: Mapping[str, str],
    opener: Callable,
) -> tuple[str, dict[str, str]]:
    token = env.get("MENDELEY_TOKEN")
    if not token:
        raise ValueError("MENDELEY_TOKEN is required for Mendeley sources")
    dataset_id = source.options.get("dataset_id")
    if not dataset_id:
        raise ValueError(f"{source.id}: missing Mendeley dataset_id")
    headers = {
        "Accept": "application/vnd.mendeley-public-dataset.1+json",
        "Authorization": f"Bearer {token}",
    }
    query = urllib.parse.urlencode({"version": source.version, "fields": "files.*"})
    request = urllib.request.Request(
        f"https://api.mendeley.com/datasets/{dataset_id}?{query}", headers=headers
    )
    with opener(request) as response:
        metadata = json.load(response)
    archives = [
        file for file in metadata.get("files", [])
        if file.get("filename", "").endswith(".zip")
    ]
    if len(archives) != 1 or not archives[0].get("id"):
        raise ValueError(f"{source.id}: expected exactly one ZIP file")
    file_id = archives[0]["id"]
    return f"https://api.mendeley.com/files/{file_id}", headers


def _download_url(
    source: SourceSpec, env: Mapping[str, str], opener: Callable
) -> tuple[str, dict[str, str]]:
    if source.resolver == "roboflow":
        return _roboflow_url(source, env, opener)
    if source.resolver == "mendeley":
        return _mendeley_url(source, env, opener)
    raise ValueError(f"{source.id}: unsupported resolver {source.resolver}")


def fetch_source(
    source: SourceSpec,
    cache_dir: Path,
    env: Mapping[str, str] = os.environ,
    *,
    opener: Callable = urllib.request.urlopen,
) -> Path:
    cache_dir = Path(cache_dir)
    if source.sha256:
        cached = cache_dir / f"{source.id}-{source.sha256[:16]}.zip"
        if cached.exists():
            digest = hashlib.sha256(cached.read_bytes()).hexdigest()
            if digest != source.sha256:
                raise ValueError(f"{source.id}: cached checksum mismatch")
            return cached

    url, headers = _download_url(source, env, opener)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{source.id}-", suffix=".part", dir=cache_dir, delete=False
    )
    temporary_path = Path(temporary.name)
    digest = hashlib.sha256()
    try:
        request = urllib.request.Request(url, headers={**headers, "Accept": "application/zip"})
        with temporary, opener(request) as response:
            while chunk := response.read(1024 * 1024):
                temporary.write(chunk)
                digest.update(chunk)
        actual = digest.hexdigest()
        if source.sha256 and actual != source.sha256:
            raise ValueError(
                f"{source.id}: checksum mismatch: expected {source.sha256}, got {actual}"
            )
        if not source.sha256:
            reviewed = cache_dir / f"{source.id}-{actual[:16]}.unverified.zip"
            temporary_path.replace(reviewed)
            raise HashReviewRequired(source.id, actual, reviewed)
        published = cache_dir / f"{source.id}-{actual[:16]}.zip"
        temporary_path.replace(published)
        return published
    except HashReviewRequired:
        raise
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class AnnotationRecord:
    category: str
    bbox: tuple[float, float, float, float]
    source_label: str


@dataclass(frozen=True)
class ImageRecord:
    id: str
    source_id: str
    source_image_id: str
    image_path: Path
    width: int
    height: int
    origin_group: str
    split_hint: str
    annotations: tuple[AnnotationRecord, ...]


@dataclass(frozen=True)
class RejectionRecord:
    source_image_id: str
    code: str
    detail: str


@dataclass(frozen=True)
class ConversionResult:
    records: tuple[ImageRecord, ...]
    rejections: tuple[RejectionRecord, ...]


def _yolo_names(root: Path) -> list[str]:
    config = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    names = config["names"]
    if isinstance(names, dict):
        return [names[key] for key in sorted(names, key=int)]
    return list(names)


def _convert_yolo(source: SourceSpec, root: Path) -> ConversionResult:
    names = _yolo_names(root)
    records = []
    rejections = []
    image_root = root / "images"
    image_paths = sorted(
        path
        for path in image_root.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    for image_path in image_paths:
        relative = image_path.relative_to(image_root)
        source_image_id = relative.as_posix()
        with Image.open(image_path) as image:
            width, height = image.size
        annotations = []
        label_path = root / "labels" / relative.with_suffix(".txt")
        lines = label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []
        for line_number, line in enumerate(lines, 1):
            try:
                class_id_text, cx_text, cy_text, width_text, height_text = line.split()
                class_id = int(class_id_text)
                cx, cy, box_width, box_height = map(
                    float, (cx_text, cy_text, width_text, height_text)
                )
                source_label = names[class_id]
                if box_width <= 0 or box_height <= 0:
                    raise ValueError("box width and height must be positive")
                if not all(0 <= value <= 1 for value in (cx, cy, box_width, box_height)):
                    raise ValueError("normalized coordinates must be within [0, 1]")
                category = source.mappings.get(source_label)
                if category is None:
                    rejections.append(
                        RejectionRecord(source_image_id, "unmapped_class", source_label)
                    )
                    continue
                bbox = (
                    (cx - box_width / 2) * width,
                    (cy - box_height / 2) * height,
                    box_width * width,
                    box_height * height,
                )
                if (
                    bbox[0] < 0
                    or bbox[1] < 0
                    or bbox[0] + bbox[2] > width
                    or bbox[1] + bbox[3] > height
                ):
                    raise ValueError("box extends outside the image")
                annotations.append(AnnotationRecord(category, bbox, source_label))
            except (ValueError, IndexError) as error:
                rejections.append(
                    RejectionRecord(
                        source_image_id, "invalid_box", f"line {line_number}: {error}"
                    )
                )
        origin_name = image_path.stem.split(".rf.")[0]
        records.append(
            ImageRecord(
                id=f"{source.id}/{source_image_id}",
                source_id=source.id,
                source_image_id=source_image_id,
                image_path=image_path,
                width=width,
                height=height,
                origin_group=f"{source.id}/{origin_name}",
                split_hint=relative.parts[0] if len(relative.parts) > 1 else "unspecified",
                annotations=tuple(annotations),
            )
        )
    return ConversionResult(tuple(records), tuple(rejections))


def _convert_coco(source: SourceSpec, root: Path) -> ConversionResult:
    records = []
    rejections = []
    for annotation_path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not {"images", "categories", "annotations"} <= data.keys():
            continue
        categories = {item["id"]: item["name"] for item in data["categories"]}
        annotations_by_image = {}
        for annotation in data["annotations"]:
            annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)
        split_hint = annotation_path.parent.name
        for image_data in sorted(data["images"], key=lambda item: str(item["file_name"])):
            source_image_id = f"{split_hint}/{image_data['file_name']}"
            image_path = annotation_path.parent / image_data["file_name"]
            width = int(image_data["width"])
            height = int(image_data["height"])
            converted = []
            for annotation in annotations_by_image.get(image_data["id"], []):
                source_label = categories.get(annotation["category_id"])
                category = source.mappings.get(source_label)
                if category is None:
                    rejections.append(
                        RejectionRecord(source_image_id, "unmapped_class", str(source_label))
                    )
                    continue
                try:
                    x, y, box_width, box_height = map(float, annotation["bbox"])
                    epsilon = 0.51
                    if (
                        box_width <= 0
                        or box_height <= 0
                        or x < -epsilon
                        or y < -epsilon
                        or x + box_width > width + epsilon
                        or y + box_height > height + epsilon
                    ):
                        raise ValueError("box is invalid or outside the image")
                    left, top = max(0.0, x), max(0.0, y)
                    right = min(float(width), x + box_width)
                    bottom = min(float(height), y + box_height)
                    converted.append(
                        AnnotationRecord(category, (left, top, right - left, bottom - top), source_label)
                    )
                except (KeyError, TypeError, ValueError) as error:
                    rejections.append(
                        RejectionRecord(source_image_id, "invalid_box", str(error))
                    )
            origin_name = image_path.stem.split(".rf.")[0]
            records.append(
                ImageRecord(
                    id=f"{source.id}/{source_image_id}",
                    source_id=source.id,
                    source_image_id=source_image_id,
                    image_path=image_path,
                    width=width,
                    height=height,
                    origin_group=f"{source.id}/{origin_name}",
                    split_hint=split_hint,
                    annotations=tuple(converted),
                )
            )
    if not records:
        raise ValueError(f"{source.id}: no COCO annotations found")
    return ConversionResult(tuple(records), tuple(rejections))


def convert_source(source: SourceSpec, extracted_dir: Path) -> ConversionResult:
    if source.options.get("format") == "yolo":
        return _convert_yolo(source, Path(extracted_dir))
    if source.options.get("format") == "coco":
        return _convert_coco(source, Path(extracted_dir))
    raise ValueError(f"{source.id}: unsupported annotation format")


from .build import BuildReport, build_dataset
