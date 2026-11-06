import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset/candidates/ppe-13c/rf-detr"
EXPECTED_CLASSES = [
    "person",
    "helmet",
    "no_helmet",
    "vest",
    "no_vest",
    "gloves",
    "no_gloves",
    "boots",
    "no_boots",
    "mask",
    "no_mask",
    "eye_protection",
    "no_eye_protection",
]
EXPECTED_IMAGES = {"train": 13764, "valid": 1713, "test": 1697}


class RFDETRSetupTest(unittest.TestCase):
    def test_dataset_is_valid_roboflow_coco_layout(self):
        for split, expected_count in EXPECTED_IMAGES.items():
            split_dir = DATASET / split
            annotation_path = split_dir / "_annotations.coco.json"
            coco = json.loads(annotation_path.read_text())

            self.assertEqual([c["name"] for c in coco["categories"]], EXPECTED_CLASSES)
            self.assertEqual(len(coco["images"]), expected_count)
            self.assertTrue(all(Path(image["file_name"]).name == image["file_name"] for image in coco["images"]))
            self.assertTrue(all((split_dir / image["file_name"]).is_file() for image in coco["images"]))

    def test_training_entrypoint_has_no_import_time_dependency(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "RF-DETR-L/train.py"), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--batch-size", result.stdout)


if __name__ == "__main__":
    unittest.main()
