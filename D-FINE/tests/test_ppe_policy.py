import unittest

from src.ppe.policy import Detection, Keypoint, PolicyConfig, evaluate_person


def visible_pose(**changes):
    points = {
        "nose": Keypoint(50, 15, 0.99),
        "left_ear": Keypoint(42, 16, 0.99),
        "right_ear": Keypoint(58, 16, 0.99),
        "left_shoulder": Keypoint(38, 32, 0.99),
        "right_shoulder": Keypoint(62, 32, 0.99),
        "left_hip": Keypoint(42, 62, 0.99),
        "right_hip": Keypoint(58, 62, 0.99),
        "left_wrist": Keypoint(28, 55, 0.99),
        "right_wrist": Keypoint(72, 55, 0.99),
        "left_ankle": Keypoint(43, 90, 0.99),
        "right_ankle": Keypoint(57, 90, 0.99),
        "left_heel": Keypoint(42, 92, 0.99),
        "right_heel": Keypoint(58, 92, 0.99),
        "left_foot_index": Keypoint(39, 94, 0.99),
        "right_foot_index": Keypoint(61, 94, 0.99),
    }
    points.update(changes)
    return points


class PPEPolicyTest(unittest.TestCase):
    def setUp(self):
        self.person = (10.0, 5.0, 90.0, 98.0)
        self.frame_size = (100, 100)
        self.config = PolicyConfig(keypoint_confidence=0.5, detection_confidence=0.5)

    def test_visible_positive_takes_precedence_over_absence(self):
        result = evaluate_person(
            self.person,
            [Detection("helmet", (35, 8, 65, 28), 0.9)],
            visible_pose(),
            self.frame_size,
            self.config,
        )

        self.assertEqual(result.states["helmet"], "present")
        self.assertEqual(result.states["no_helmet"], "absent")

    def test_visible_region_without_positive_emits_no_label(self):
        result = evaluate_person(
            self.person, [], visible_pose(), self.frame_size, self.config
        )

        self.assertEqual(result.states["helmet"], "absent")
        self.assertEqual(result.states["no_helmet"], "present")
        self.assertEqual(result.states["no_vest"], "present")
        self.assertEqual(result.states["no_gloves"], "present")
        self.assertEqual(result.states["no_boots"], "present")

    def test_low_confidence_body_region_is_unknown_not_negative(self):
        pose = visible_pose(
            left_wrist=Keypoint(28, 55, 0.1),
            right_wrist=Keypoint(72, 55, 0.1),
        )

        result = evaluate_person(self.person, [], pose, self.frame_size, self.config)

        self.assertEqual(result.states["gloves"], "unknown")
        self.assertEqual(result.states["no_gloves"], "unknown")

    def test_one_visible_unprotected_hand_proves_no_gloves(self):
        pose = visible_pose(right_wrist=Keypoint(72, 55, 0.1))

        result = evaluate_person(self.person, [], pose, self.frame_size, self.config)

        self.assertEqual(result.states["gloves"], "absent")
        self.assertEqual(result.states["no_gloves"], "present")

    def test_visible_glove_on_one_hand_with_other_hand_unknown_stays_unknown(self):
        pose = visible_pose(right_wrist=Keypoint(72, 55, 0.1))
        detections = [Detection("gloves", (23, 50, 33, 60), 0.9)]

        result = evaluate_person(
            self.person, detections, pose, self.frame_size, self.config
        )

        self.assertEqual(result.states["gloves"], "unknown")
        self.assertEqual(result.states["no_gloves"], "unknown")


if __name__ == "__main__":
    unittest.main()
