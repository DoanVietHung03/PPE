import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.solver._solver import BaseSolver


class CheckpointLoadingTest(unittest.TestCase):
    def test_local_checkpoint_uses_weights_only_loader(self):
        solver = BaseSolver(SimpleNamespace())
        solver.load_state_dict = Mock()

        with patch("src.solver._solver.torch.load", return_value={}) as load:
            solver.load_resume_state("checkpoint.pth")

        load.assert_called_once_with("checkpoint.pth", map_location="cpu", weights_only=True)


if __name__ == "__main__":
    unittest.main()
