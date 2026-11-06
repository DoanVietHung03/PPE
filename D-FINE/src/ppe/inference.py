from .policy import Detection, Keypoint


DETECTOR_CLASSES = ("person", "helmet", "vest", "gloves", "boots")
POSE_LANDMARKS = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
)


def padded_crop(box, frame_size, padding=0.1):
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    frame_width, frame_height = frame_size
    return (
        max(0, int(x1 - width * padding)),
        max(0, int(y1 - height * padding)),
        min(frame_width, int(x2 + width * padding + 0.999)),
        min(frame_height, int(y2 + height * padding + 0.999)),
    )


def map_pose_landmarks(landmarks, crop_box):
    if len(landmarks) != len(POSE_LANDMARKS):
        raise ValueError(f"expected {len(POSE_LANDMARKS)} pose landmarks")
    left, top, right, bottom = crop_box
    width, height = right - left, bottom - top
    return {
        name: Keypoint(
            left + landmark.x * width,
            top + landmark.y * height,
            min(float(landmark.visibility), float(landmark.presence)),
        )
        for name, landmark in zip(POSE_LANDMARKS, landmarks)
    }


def detections_from_outputs(labels, boxes, scores, threshold=0.5):
    return [
        Detection(DETECTOR_CLASSES[int(label)], tuple(map(float, box)), float(score))
        for label, box, score in zip(labels, boxes, scores)
        if float(score) >= threshold and 0 <= int(label) < len(DETECTOR_CLASSES)
    ]


def filter_detections(detections, thresholds):
    return [
        item for item in detections if item.score >= thresholds[item.category]
    ]


def dataset_image_paths(coco, root, limit=None):
    images = coco["images"][:limit]
    return [
        (image["id"], image["file_name"], root / image["file_name"])
        for image in images
    ]
