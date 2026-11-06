import unittest

from src.ppe.evaluation import coco_state_samples, evaluate_states


class PPEEvaluationTest(unittest.TestCase):
    def test_builds_person_state_samples_from_coco_and_predictions(self):
        instances = {
            "categories": [{"id": 0, "name": "person"}],
            "annotations": [
                {"image_id": 7, "category_id": 0, "bbox": [10, 10, 40, 80]}
            ],
        }
        states = {
            "categories": [
                {"id": 0, "name": "helmet"},
                {"id": 4, "name": "no_helmet"},
            ],
            "annotations": [
                {"image_id": 7, "category_id": 4, "bbox": [20, 12, 10, 10]},
                {"image_id": 8, "category_id": 4, "bbox": [20, 12, 10, 10]},
            ],
        }
        predictions = [{
            "image_id": 7,
            "people": [{
                "box": [11, 10, 51, 90],
                "states": {"helmet": "absent", "no_helmet": "present"},
            }],
        }]

        samples = coco_state_samples(instances, states, predictions)

        self.assertEqual(samples, [{
            "ground_truth": {"helmet": "absent", "no_helmet": "present"},
            "prediction": {"helmet": "absent", "no_helmet": "present"},
            "buckets": {},
        }])

    def test_counts_unknown_prediction_as_missed_positive_and_reports_coverage(self):
        report = evaluate_states(
            [
                {
                    "ground_truth": {"helmet": "present", "no_helmet": "absent"},
                    "prediction": {"helmet": "present", "no_helmet": "absent"},
                    "buckets": {"image_kind": "full_frame"},
                },
                {
                    "ground_truth": {"helmet": "absent", "no_helmet": "present"},
                    "prediction": {"helmet": "absent", "no_helmet": "unknown"},
                    "buckets": {"image_kind": "person_crop"},
                },
            ]
        )

        self.assertEqual(report["labels"]["helmet"]["f1"], 1.0)
        self.assertEqual(report["labels"]["no_helmet"]["false_negatives"], 1)
        self.assertEqual(report["labels"]["no_helmet"]["unknown_coverage"], 0.5)

    def test_counts_present_prediction_on_absent_ground_truth_as_false_alarm(self):
        report = evaluate_states(
            [
                {
                    "ground_truth": {"vest": "absent", "no_vest": "present"},
                    "prediction": {"vest": "present", "no_vest": "absent"},
                    "buckets": {"source": "camera-a"},
                }
            ]
        )

        self.assertEqual(report["labels"]["vest"]["false_positives"], 1)
        self.assertEqual(report["false_alarms_per_person"], 1.0)
        self.assertEqual(report["buckets"]["source=camera-a"]["people"], 1)

    def test_ignores_unknown_ground_truth_for_precision_and_recall(self):
        report = evaluate_states(
            [
                {
                    "ground_truth": {"boots": "unknown", "no_boots": "unknown"},
                    "prediction": {"boots": "present", "no_boots": "absent"},
                    "buckets": {},
                }
            ]
        )

        self.assertEqual(report["labels"]["boots"]["evaluated"], 0)
        self.assertIsNone(report["labels"]["boots"]["f1"])


    def test_unknown_prediction_is_not_a_true_negative(self):
        report = evaluate_states([{
            "ground_truth": {"boots": "absent"},
            "prediction": {"boots": "unknown"},
            "buckets": {},
        }])

        self.assertEqual(report["labels"]["boots"]["true_negatives"], 0)
        self.assertEqual(report["labels"]["boots"]["evaluated"], 0)

    def test_reports_metrics_inside_each_bucket(self):
        report = evaluate_states([{
            "ground_truth": {"helmet": "present"},
            "prediction": {"helmet": "present"},
            "buckets": {"image_kind": "full_frame"},
        }])

        self.assertEqual(report["buckets"]["image_kind=full_frame"]["macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
