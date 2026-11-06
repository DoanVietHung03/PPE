import unittest
from types import SimpleNamespace

from src.ppe.build import _assign_splits


class GroupSplitTest(unittest.TestCase):
    def test_balances_rare_classes_and_sources_across_grouped_splits(self):
        records = [
            SimpleNamespace(
                id=f"record-{index}",
                source_id="a" if index % 2 else "b",
                annotations=(SimpleNamespace(category="boots"),) if index < 10 else (),
            )
            for index in range(30)
        ]

        assignments = _assign_splits(records, {index: [index] for index in range(30)}, seed=0)

        boots = [sum(assignments[index] == split for index in range(10)) for split in ("train", "val", "test")]
        source_a = [sum(assignments[index] == split and records[index].source_id == "a" for index in range(30)) for split in ("train", "val", "test")]
        self.assertEqual(boots, [8, 1, 1])
        self.assertEqual(source_a[0], 12)
        self.assertEqual(sorted(source_a[1:]), [1, 2])


if __name__ == "__main__":
    unittest.main()
