import unittest
from pathlib import Path
from types import SimpleNamespace

from src.ppe.inference import dataset_image_paths, detections_from_outputs, filter_detections, map_pose_landmarks, padded_crop


class PPEInferenceAdapterTest(unittest.TestCase):
    def test_maps_normalized_crop_landmarks_back_to_frame(self):
        landmarks = [SimpleNamespace(x=0.5, y=0.25, visibility=0.8, presence=0.6)] * 33

        mapped = map_pose_landmarks(landmarks, (10, 20, 110, 220))

        self.assertEqual((mapped["nose"].x, mapped["nose"].y), (60.0, 70.0))
        self.assertEqual(mapped["nose"].score, 0.6)

    def test_maps_detector_outputs_and_filters_confidence(self):
        detections = detections_from_outputs(
            [0, 3, 4], [[1, 2, 11, 22], [3, 4, 13, 24], [5, 6, 15, 26]],
            [0.9, 0.7, 0.2], threshold=0.5,
        )

        self.assertEqual([item.category for item in detections], ["person", "gloves"])

    def test_person_crop_padding_is_bounded_by_frame(self):
        self.assertEqual(padded_crop((0, 5, 80, 95), (100, 100)), (0, 0, 88, 100))
    def test_lists_coco_images_with_optional_limit(self):
        coco = {"images": [
            {"id": 7, "file_name": "images/test/a.jpg"},
            {"id": 8, "file_name": "images/test/b.jpg"},
        ]}

        self.assertEqual(
            dataset_image_paths(coco, Path("/dataset"), limit=1),
            [(7, "images/test/a.jpg", Path("/dataset/images/test/a.jpg"))],
        )


    def test_filters_each_detection_with_its_class_threshold(self):
        detections = detections_from_outputs(
            [0, 1, 2], [[0, 0, 10, 10]] * 3, [0.39, 0.41, 0.24], threshold=0.0
        )
        filtered = filter_detections(
            detections, {"person": 0.4, "helmet": 0.4, "vest": 0.25}
        )

        self.assertEqual([item.category for item in filtered], ["helmet"])


if __name__ == "__main__":
    unittest.main()
