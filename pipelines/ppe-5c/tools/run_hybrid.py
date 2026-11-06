import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
DFINE_ROOT = ROOT / "D-FINE"
sys.path.insert(0, str(DFINE_ROOT))

from src.core import YAMLConfig
from src.ppe.inference import dataset_image_paths, detections_from_outputs, filter_detections, map_pose_landmarks, padded_crop
from src.ppe.policy import Detection, PolicyConfig, evaluate_person
from src.ppe.registry import load_registry

DEFAULT_THRESHOLDS = {
    "person": 0.40,
    "helmet": 0.40,
    "vest": 0.23,
    "gloves": 0.22,
    "boots": 0.22,
}



class Detector:
    def __init__(self, config, checkpoint, device, input_size):
        cfg = YAMLConfig(str(config), resume=str(checkpoint))
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = saved["ema"]["module"] if "ema" in saved else saved["model"]
        cfg.model.load_state_dict(state)
        self.model = cfg.model.deploy().to(device).eval()
        self.postprocessor = cfg.postprocessor.deploy().to(device).eval()
        self.device = device
        self.input_size = input_size

    def __call__(self, image, thresholds):
        width, height = image.size
        array = np.asarray(image.resize((self.input_size, self.input_size)), dtype=np.float32)
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(self.device) / 255
        size = torch.tensor([[width, height]], device=self.device)
        with torch.inference_mode():
            labels, boxes, scores = self.postprocessor(self.model(tensor), size)
        detections = detections_from_outputs(
            labels[0].cpu().tolist(), boxes[0].cpu().tolist(), scores[0].cpu().tolist(), min(thresholds.values())
        )
        return filter_detections(detections, thresholds)


def process_frame(image, detector, pose, args):
    width, height = image.size
    thresholds = {
        name: args.detection_confidence if args.detection_confidence is not None else getattr(args, f"{name}_confidence")
        for name in DEFAULT_THRESHOLDS
    }
    detections = detector(image, thresholds)
    people = [detection for detection in detections if detection.category == "person"]
    if args.person_crop:
        people = [Detection("person", (0, 0, width, height), 1.0)]
    results = []
    policy = PolicyConfig(
        keypoint_confidence=args.keypoint_confidence,
        detection_confidence=0.0,
        region_padding=args.region_padding,
    )
    for person in people:
        crop_box = padded_crop(person.bbox, (width, height), args.crop_padding)
        crop = np.ascontiguousarray(np.asarray(image.crop(crop_box).convert("RGB")))
        pose_result = pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
        keypoints = map_pose_landmarks(pose_result.pose_landmarks[0], crop_box) if pose_result.pose_landmarks else {}
        state = evaluate_person(person.bbox, detections, keypoints, (width, height), policy)
        results.append({
            "box": list(person.bbox),
            "score": person.score,
            "labels": [name for name, value in state.states.items() if value == "present"],
            "states": state.states,
            "keypoints": {name: {"x": point.x, "y": point.y, "score": point.score} for name, point in keypoints.items()},
        })
    return {
        "size": [width, height],
        "detections": [detection.__dict__ for detection in detections],
        "people": results,
    }


def render(image, result):
    drawing = ImageDraw.Draw(image)
    for person in result["people"]:
        drawing.rectangle(person["box"], outline="yellow", width=3)
        drawing.text((person["box"][0], person["box"][1]), ", ".join(person["labels"]), fill="yellow")
    return image


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description="D-FINE + visibility-gated BlazePose PPE inference")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DFINE_ROOT / "configs/ppe/sources.yml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-media", type=Path)
    parser.add_argument("--person-crop", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size", type=int, default=800)
    parser.add_argument("--detection-confidence", type=float)
    for name, default in DEFAULT_THRESHOLDS.items():
        parser.add_argument(f"--{name}-confidence", type=float, default=default)
    parser.add_argument("--keypoint-confidence", type=float, default=0.5)
    parser.add_argument("--region-padding", type=float, default=0.1)
    parser.add_argument("--crop-padding", type=float, default=0.1)
    args = parser.parse_args(argv)

    pose_spec = load_registry(args.registry).require_approved_model("blazepose_heavy")
    pose_hash = sha256(args.pose_model)
    if pose_hash != pose_spec.sha256:
        raise ValueError(f"pose model checksum mismatch: expected {pose_spec.sha256}, got {pose_hash}")
    detector = Detector(args.config, args.checkpoint, args.device, args.input_size)
    pose_options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.pose_model)), num_poses=1
    )
    metadata = {
        "checkpoint_sha256": sha256(args.checkpoint),
        "pose_model_sha256": pose_hash,
        "pose_model_license": pose_spec.license,
        "thresholds": {
            "detection": {
                name: args.detection_confidence if args.detection_confidence is not None else getattr(args, f"{name}_confidence")
                for name in DEFAULT_THRESHOLDS
            },
            "keypoint": args.keypoint_confidence,
            "region_padding": args.region_padding,
            "crop_padding": args.crop_padding,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    with mp.tasks.vision.PoseLandmarker.create_from_options(pose_options) as pose:
        if args.input.suffix.lower() == ".json":
            if args.output_media:
                raise ValueError("--output-media is unavailable for dataset input")
            coco = json.loads(args.input.read_text(encoding="utf-8"))
            root = args.dataset_root or args.input.parents[1]
            with args.output_json.open("w", encoding="utf-8") as output:
                for image_id, file_name, path in dataset_image_paths(coco, root, args.limit):
                    image = Image.open(path).convert("RGB")
                    result = {
                        "image_id": image_id, "file_name": file_name,
                        **metadata, **process_frame(image, detector, pose, args),
                    }
                    output.write(json.dumps(result, separators=(",", ":")) + "\n")
            return 0
        if args.input.suffix.lower() in image_suffixes:
            image = Image.open(args.input).convert("RGB")
            result = {**metadata, **process_frame(image, detector, pose, args)}
            args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            if args.output_media:
                render(image, result).save(args.output_media)
            return 0

        capture = cv2.VideoCapture(str(args.input))
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {args.input}")
        writer = None
        if args.output_media:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS) or 25
            writer = cv2.VideoWriter(str(args.output_media), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        with args.output_json.open("w", encoding="utf-8") as output:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = {"frame": frame_index, **metadata, **process_frame(image, detector, pose, args)}
                output.write(json.dumps(result, separators=(",", ":")) + "\n")
                if writer:
                    rendered = cv2.cvtColor(np.asarray(render(image, result)), cv2.COLOR_RGB2BGR)
                    writer.write(rendered)
                frame_index += 1
        capture.release()
        if writer:
            writer.release()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
