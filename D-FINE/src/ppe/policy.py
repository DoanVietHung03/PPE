from dataclasses import dataclass
from typing import Literal, Mapping


State = Literal["present", "absent", "unknown"]
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    score: float


@dataclass(frozen=True)
class Detection:
    category: str
    bbox: Box
    score: float


@dataclass(frozen=True)
class PolicyConfig:
    keypoint_confidence: float = 0.5
    detection_confidence: float = 0.5
    frame_margin: float = 2.0
    region_padding: float = 0.1


@dataclass(frozen=True)
class PersonPPEState:
    states: dict[str, State]


def _inside(point: Keypoint, frame_size, config: PolicyConfig) -> bool:
    width, height = frame_size
    return (
        point.score >= config.keypoint_confidence
        and config.frame_margin <= point.x <= width - config.frame_margin
        and config.frame_margin <= point.y <= height - config.frame_margin
    )


def _region(
    keypoints: Mapping[str, Keypoint],
    names: tuple[str, ...],
    frame_size,
    person_box: Box,
    config: PolicyConfig,
) -> Box | None:
    points = [keypoints.get(name) for name in names]
    if any(point is None or not _inside(point, frame_size, config) for point in points):
        return None
    person_width = person_box[2] - person_box[0]
    person_height = person_box[3] - person_box[1]
    pad_x = max(3.0, person_width * config.region_padding)
    pad_y = max(3.0, person_height * config.region_padding)
    return (
        min(point.x for point in points) - pad_x,
        min(point.y for point in points) - pad_y,
        max(point.x for point in points) + pad_x,
        max(point.y for point in points) + pad_y,
    )


def _center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _contains(box: Box, point: tuple[float, float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _states_for(
    positive: str,
    regions: tuple[Box | None, ...],
    detections: list[Detection],
) -> tuple[State, State]:
    visible = [region for region in regions if region is not None]
    if not visible:
        return "unknown", "unknown"
    matched = [
        any(_contains(region, _center(detection.bbox)) for detection in detections)
        for region in visible
    ]
    if not all(matched):
        return "absent", "present"
    if len(visible) == len(regions):
        return "present", "absent"
    return "unknown", "unknown"


def evaluate_person(
    person_box: Box,
    detections: list[Detection],
    keypoints: Mapping[str, Keypoint],
    frame_size: tuple[int, int],
    config: PolicyConfig = PolicyConfig(),
) -> PersonPPEState:
    detections_by_class = {
        category: [
            detection
            for detection in detections
            if detection.category == category
            and detection.score >= config.detection_confidence
            and _contains(person_box, _center(detection.bbox))
        ]
        for category in ("helmet", "vest", "gloves", "boots")
    }
    regions = {
        "helmet": (
            _region(
                keypoints,
                ("nose", "left_ear", "right_ear"),
                frame_size,
                person_box,
                config,
            ),
        ),
        "vest": (
            _region(
                keypoints,
                ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
                frame_size,
                person_box,
                config,
            ),
        ),
        "gloves": tuple(
            _region(keypoints, (f"{side}_wrist",), frame_size, person_box, config)
            for side in ("left", "right")
        ),
        "boots": tuple(
            _region(
                keypoints,
                (f"{side}_ankle", f"{side}_heel", f"{side}_foot_index"),
                frame_size,
                person_box,
                config,
            )
            for side in ("left", "right")
        ),
    }
    states: dict[str, State] = {}
    for positive in ("helmet", "vest", "gloves", "boots"):
        positive_state, negative_state = _states_for(
            positive, regions[positive], detections_by_class[positive]
        )
        states[positive] = positive_state
        states[f"no_{positive}"] = negative_state
    return PersonPPEState(states)
