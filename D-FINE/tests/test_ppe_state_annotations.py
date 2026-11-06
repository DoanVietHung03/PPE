import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.ppe.dataset import AnnotationRecord, ImageRecord, build_dataset


class StateAnnotationBuildTest(unittest.TestCase):
    def test_preserves_explicit_negative_labels_without_inventing_missing_ones(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "person.jpg"
            Image.new("RGB", (100, 100), "white").save(image)
            record = ImageRecord(
                "source/person", "source", "person.jpg", image, 100, 100,
                "source/origin", "train",
                (
                    AnnotationRecord("person", (10, 5, 80, 90), "Person"),
                    AnnotationRecord("no_helmet", (35, 10, 30, 20), "NoHelmet"),
                ),
            )

            report = build_dataset([record], root / "output", seed=5)
            states = json.loads((report.path / "annotations" / "states_train.json").read_text())
            names = {category["id"]: category["name"] for category in states["categories"]}
            labels = [names[item["category_id"]] for item in states["annotations"]]

            self.assertIn("no_helmet", labels)
            self.assertNotIn("no_gloves", labels)


if __name__ == "__main__":
    unittest.main()
