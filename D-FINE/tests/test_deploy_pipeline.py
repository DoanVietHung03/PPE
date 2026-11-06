import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[2] / "pipelines" / "ppe-5c"))

from deploy.pipeline import (
    filter_detections,
    open_video_writer,
    preprocess,
    process_frame,
    render,
    thresholds_from_args,
    unit_float,
    validate_paths,
)
from deploy.ppe.policy import Detection


def test_preprocess_matches_onnx_contract():
    image = Image.new("RGB", (4, 2), (255, 128, 0))

    images, sizes = preprocess(image)

    assert images.shape == (1, 3, 960, 960)
    assert images.dtype == np.float32
    assert sizes.tolist() == [[4, 2]]
    assert sizes.dtype == np.int64
    assert images.min() >= 0.0
    assert images.max() <= 1.0


def test_output_paths_must_be_distinct(tmp_path):
    input_path = tmp_path / "input.mp4"

    try:
        validate_paths(input_path, input_path, None)
    except ValueError as error:
        assert "distinct" in str(error)
    else:
        raise AssertionError("path collision was accepted")


def test_confidence_must_be_between_zero_and_one():
    assert unit_float("0.5") == 0.5
    for value in ("-0.1", "1.1", "nan"):
        try:
            unit_float(value)
        except argparse.ArgumentTypeError:
            pass
        else:
            raise AssertionError(f"invalid confidence accepted: {value}")


def test_video_writer_failure_is_reported(monkeypatch, tmp_path):
    class ClosedWriter:
        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr("deploy.pipeline.cv2.VideoWriter", lambda *args: ClosedWriter())

    try:
        open_video_writer(tmp_path / "output.mp4", 25.0, (640, 480))
    except ValueError as error:
        assert "video writer" in str(error)


def test_render_draws_detection_without_person():
    image = Image.new("RGB", (100, 100), "black")
    result = {
        "detections": [
            {"category": "helmet", "bbox": [10, 10, 30, 30], "score": 0.8}
        ],
        "people": [],
    }

    render(image, result)

    assert image.getpixel((10, 10)) == (0, 191, 255)


def test_render_marks_person_red_when_ppe_is_absent():
    image = Image.new("RGB", (100, 100), "black")
    states = {
        "helmet": "absent",
        "no_helmet": "present",
        "vest": "present",
        "no_vest": "absent",
        "gloves": "unknown",
        "no_gloves": "unknown",
        "boots": "present",
        "no_boots": "absent",
    }
    result = {
        "detections": [{"category": "person", "bbox": [10, 10, 90, 90], "score": 0.9}],
        "people": [{"box": [10, 10, 90, 90], "states": states}],
    }

    render(image, result)

    assert image.getpixel((10, 10)) == (255, 0, 0)


def test_render_does_not_write_detection_text_inside_ppe_box():
    image = Image.new("RGB", (240, 240), "black")
    result = {
        "detections": [
            {"category": "helmet", "bbox": [100, 100, 180, 180], "score": 0.8}
        ],
        "people": [],
    }

    render(image, result)

    assert all(
        image.getpixel((x, y)) == (0, 0, 0)
        for x in range(104, 176)
        for y in range(104, 116)
    )


def test_render_keeps_person_status_text_inside_person_box():
    image = Image.new("RGB", (300, 120), "black")
    states = {
        "helmet": "present",
        "vest": "absent",
        "gloves": "unknown",
        "boots": "present",
    }
    result = {
        "detections": [],
        "people": [{"box": [10, 10, 80, 100], "states": states}],
    }

    render(image, result)

    assert all(
        image.getpixel((x, y)) == (0, 0, 0)
        for x in range(81, image.width)
        for y in range(image.height)
    )


def test_detections_use_threshold_for_their_class():
    detections = [
        Detection("person", (0, 0, 1, 1), 0.39),
        Detection("person", (0, 0, 1, 1), 0.40),
        Detection("vest", (0, 0, 1, 1), 0.23),
        Detection("boots", (0, 0, 1, 1), 0.21),
        Detection("boots", (0, 0, 1, 1), 0.22),
    ]
    thresholds = {
        "person": 0.40,
        "helmet": 0.40,
        "vest": 0.23,
        "gloves": 0.22,
        "boots": 0.22,
    }

    kept = filter_detections(detections, thresholds)

    assert [(item.category, item.score) for item in kept] == [
        ("person", 0.40),
        ("vest", 0.23),
        ("boots", 0.22),
    ]


def test_global_threshold_overrides_class_thresholds():
    args = SimpleNamespace(
        detection_confidence=0.5,
        person_confidence=0.4,
        helmet_confidence=0.4,
        vest_confidence=0.23,
        gloves_confidence=0.22,
        boots_confidence=0.22,
    )

    assert set(thresholds_from_args(args).values()) == {0.5}


def test_process_frame_retries_pose_with_another_crop_padding():
    landmark = SimpleNamespace(x=0.5, y=0.5, visibility=1.0, presence=1.0)

    class Pose:
        def __init__(self):
            self.results = [
                SimpleNamespace(pose_landmarks=[]),
                SimpleNamespace(pose_landmarks=[[landmark] * 33]),
            ]

        def detect(self, image):
            return self.results.pop(0)

    class Detector:
        def __call__(self, image, thresholds):
            return [Detection("person", (20, 10, 80, 90), 0.9)]

    args = SimpleNamespace(
        detection_confidence=None,
        person_confidence=0.4,
        helmet_confidence=0.4,
        vest_confidence=0.23,
        gloves_confidence=0.22,
        boots_confidence=0.22,
        keypoint_confidence=0.5,
        region_padding=0.1,
        crop_padding=0.25,
    )

    result = process_frame(Image.new("RGB", (100, 100)), Detector(), Pose(), args)

    assert len(result["people"][0]["keypoints"]) == 33
