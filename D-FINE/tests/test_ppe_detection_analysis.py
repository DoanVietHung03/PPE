import json
import tempfile
import unittest
from pathlib import Path

from tools.ppe.analyze_detection import load_full_frame_test_set, match, parse_args


class PPEDetectionAnalysisTest(unittest.TestCase):
    def test_matching_prioritizes_prediction_confidence(self):
        ground_truth = [("helmet", (0, 0, 10, 10))]
        predictions = [
            ("helmet", (0, 0, 8, 10), 0.9),
            ("helmet", (0, 0, 10, 10), 0.6),
        ]
        matched_ground_truth, matched_predictions, _ = match(
            ground_truth, predictions, score_threshold=0.5
        )

        self.assertEqual(matched_ground_truth, {0})
        self.assertEqual(matched_predictions, {0})

    def test_cli_accepts_dataset_checkpoint_and_output(self):
        args = parse_args([
            "--dataset", "dataset/build",
            "--checkpoint", "model.pth",
            "--output", "report",
        ])

        self.assertEqual(args.dataset, Path("dataset/build"))
        self.assertEqual(args.checkpoint, Path("model.pth"))
        self.assertEqual(args.output, Path("report"))

    def test_loads_only_full_frame_test_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "annotations").mkdir()
            (root / "manifest.csv").write_text(
                "source_id,split,image_kind,output_image\n"
                "camera,test,full_frame,images/test/a.jpg\n"
                "camera,test,person_crop,images/test/b.jpg\n"
                "camera,train,full_frame,images/train/c.jpg\n",
                encoding="utf-8",
            )
            (root / "annotations/instances_test.json").write_text(json.dumps({
                "categories": [{"id": 0, "name": "person"}],
                "images": [
                    {"id": 1, "file_name": "images/test/a.jpg"},
                    {"id": 2, "file_name": "images/test/b.jpg"},
                ],
                "annotations": [],
            }), encoding="utf-8")

            images, _, _, manifest = load_full_frame_test_set(root)

        self.assertEqual([image["id"] for image in images], [1])
        self.assertEqual(manifest["images/test/a.jpg"]["source_id"], "camera")


if __name__ == "__main__":
    unittest.main()
