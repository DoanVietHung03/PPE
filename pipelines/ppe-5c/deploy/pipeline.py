import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

try:
    from .ppe.inference import detections_from_outputs, map_pose_landmarks, padded_crop
    from .ppe.policy import PolicyConfig, evaluate_person
except ImportError:
    from ppe.inference import detections_from_outputs, map_pose_landmarks, padded_crop
    from ppe.policy import PolicyConfig, evaluate_person


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_THRESHOLDS = {
    "person": 0.40,
    "helmet": 0.40,
    "vest": 0.23,
    "gloves": 0.22,
    "boots": 0.22,
}


def preprocess(image):
    width, height = image.size
    resized = image.resize((960, 960), Image.Resampling.BILINEAR)
    images = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(images), np.array([[width, height]], dtype=np.int64)


def unit_float(value):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be a finite number between 0 and 1")
    return number


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return number


def thresholds_from_args(args):
    if args.detection_confidence is not None:
        return dict.fromkeys(DEFAULT_THRESHOLDS, args.detection_confidence)
    return {name: getattr(args, f"{name}_confidence") for name in DEFAULT_THRESHOLDS}


def filter_detections(detections, thresholds):
    return [
        detection for detection in detections
        if detection.score >= thresholds[detection.category]
    ]


def validate_paths(input_path, output_json, output_media):
    paths = [input_path.resolve(), output_json.resolve()]
    if output_media:
        paths.append(output_media.resolve())
    if len(paths) != len(set(paths)):
        raise ValueError("input and output paths must be distinct")


def open_video_writer(path, fps, size):
    width, height = size
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("invalid input video metadata")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        writer.release()
        raise ValueError(f"cannot open video writer: {path}")
    return writer


class OnnxDetector:
    def __init__(self, model_path):
        ort.preload_dlls()
        self.session = ort.InferenceSession(
            str(model_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        if self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("CUDAExecutionProvider is unavailable; refusing CPU fallback")

    def __call__(self, image, thresholds):
        images, sizes = preprocess(image)
        labels, boxes, scores = self.session.run(
            None, {"images": images, "orig_target_sizes": sizes}
        )
        detections = detections_from_outputs(
            labels[0], boxes[0], scores[0], min(thresholds.values())
        )
        return filter_detections(detections, thresholds)


def process_frame(image, detector, pose, args):
    width, height = image.size
    thresholds = thresholds_from_args(args)
    detections = detector(image, thresholds)
    people = [item for item in detections if item.category == "person"]
    policy = PolicyConfig(
        keypoint_confidence=args.keypoint_confidence,
        detection_confidence=0.0,
        region_padding=args.region_padding,
    )
    results = []
    for person in people:
        keypoints = {}
        valid_crop = False
        for padding in dict.fromkeys((args.crop_padding, 0.10, 0.25, 0.40)):
            crop_box = padded_crop(person.bbox, (width, height), padding)
            if crop_box[0] >= crop_box[2] or crop_box[1] >= crop_box[3]:
                continue
            valid_crop = True
            crop = np.ascontiguousarray(np.asarray(image.crop(crop_box).convert("RGB")))
            pose_result = pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
            if pose_result.pose_landmarks:
                keypoints = map_pose_landmarks(pose_result.pose_landmarks[0], crop_box)
                break
        if not valid_crop:
            continue
        state = evaluate_person(person.bbox, detections, keypoints, (width, height), policy)
        results.append(
            {
                "box": list(person.bbox),
                "score": person.score,
                "states": state.states,
                "keypoints": {
                    name: {"x": point.x, "y": point.y, "score": point.score}
                    for name, point in keypoints.items()
                },
            }
        )
    return {
        "size": [width, height],
        "thresholds": thresholds,
        "detections": [item.__dict__ for item in detections],
        "people": results,
    }


def render(image, result):
    drawing = ImageDraw.Draw(image)
    colors = {
        "helmet": "deepskyblue",
        "vest": "lime",
        "gloves": "orange",
        "boots": "magenta",
    }
    for detection in result["detections"]:
        color = colors.get(detection["category"])
        if color is None:
            continue
        drawing.rectangle(detection["bbox"], outline=color, width=3)

    legend = list(colors.items())
    drawing.rectangle((4, 4, 74, 8 + 12 * len(legend)), fill="black")
    for index, (name, color) in enumerate(legend):
        y = 7 + index * 12
        drawing.rectangle((7, y + 2, 13, y + 8), fill=color)
        drawing.text((17, y), name, fill=color)
    for person in result["people"]:
        states = person["states"]
        equipment = ("helmet", "vest", "gloves", "boots")
        color = "red" if any(states[name] == "absent" for name in equipment) else (
            "orange" if any(states[name] == "unknown" for name in equipment) else "lime"
        )
        drawing.rectangle(person["box"], outline=color, width=3)
        symbols = {"present": "OK", "absent": "NO", "unknown": "?"}
        labels = [f"{name[0].upper()}:{symbols[states[name]]}" for name in equipment]
        x, y = int(person["box"][0]) + 4, int(person["box"][1]) + 4
        line_height = 11
        width = max(drawing.textbbox((0, 0), label)[2] for label in labels)
        drawing.rectangle((x - 2, y - 2, x + width + 2, y + line_height * 4), fill="black")
        for index, label in enumerate(labels):
            drawing.text((x, y + index * line_height), label, fill=color)
    return image


def run_image(args, detector, pose):
    image = Image.open(args.input).convert("RGB")
    result = process_frame(image, detector, pose, args)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.output_media:
        args.output_media.parent.mkdir(parents=True, exist_ok=True)
        render(image, result).save(args.output_media)


def run_video(args, detector, pose):
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise ValueError(f"cannot open input: {args.input}")
    writer = None
    if args.output_media:
        args.output_media.parent.mkdir(parents=True, exist_ok=True)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        writer = open_video_writer(args.output_media, fps, (width, height))
    try:
        with args.output_json.open("w", encoding="utf-8") as output:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = {"frame": frame_index, **process_frame(image, detector, pose, args)}
                output.write(json.dumps(result, separators=(",", ":")) + "\n")
                if writer:
                    rendered = cv2.cvtColor(np.asarray(render(image, result)), cv2.COLOR_RGB2BGR)
                    writer.write(rendered)
                frame_index += 1
    finally:
        capture.release()
        if writer:
            writer.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description="ONNX D-FINE + MediaPipe PPE pipeline")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-json", type=Path, required=True, help="JSON for images; JSONL for videos"
    )
    parser.add_argument("--output-media", type=Path)
    parser.add_argument("--detector", type=Path, default=ROOT / "models/dfine_m_960.onnx")
    parser.add_argument(
        "--pose-model", type=Path, default=ROOT / "models/pose_landmarker_heavy.task"
    )
    parser.add_argument(
        "--detection-confidence", type=unit_float, help="override every class threshold"
    )
    for name, default in DEFAULT_THRESHOLDS.items():
        parser.add_argument(f"--{name}-confidence", type=unit_float, default=default)
    parser.add_argument("--keypoint-confidence", type=unit_float, default=0.5)
    parser.add_argument("--region-padding", type=nonnegative_float, default=0.1)
    parser.add_argument("--crop-padding", type=nonnegative_float, default=0.1)
    args = parser.parse_args(argv)

    validate_paths(args.input, args.output_json, args.output_media)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    detector = OnnxDetector(args.detector)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.pose_model)), num_poses=1
    )
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as pose:
        if args.input.suffix.lower() in IMAGE_SUFFIXES:
            run_image(args, detector, pose)
        else:
            run_video(args, detector, pose)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
