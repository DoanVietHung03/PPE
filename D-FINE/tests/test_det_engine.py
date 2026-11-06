import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.solver.det_engine import _validate_samples


class ValidateSamplesTest(unittest.TestCase):
    def test_reports_nonfinite_batch_and_image_ids(self):
        samples = torch.zeros(2, 3, 4, 4)
        samples[1, 2, 3, 1] = torch.nan
        targets = [{"image_id": torch.tensor(17)}, {"image_id": torch.tensor(42)}]

        with self.assertRaisesRegex(
            RuntimeError,
            r"epoch=16, batch=501, image_ids=\[17, 42\], first_bad=\[1, 2, 3, 1\]",
        ):
            _validate_samples(samples, targets, epoch=16, batch=501)


if __name__ == "__main__":
    unittest.main()
