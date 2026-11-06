import unittest
from pathlib import Path


ROOT = Path(__file__).parents[2]


class PPELayoutTest(unittest.TestCase):
    def test_canonical_pipeline_files_exist(self):
        expected = (
            "pipelines/ppe-5c/configs/dfine_m_960.yml",
            "pipelines/ppe-5c/deploy/pipeline.py",
            "pipelines/ppe-5c/run-pipeline.sh",
            "pipelines/ppe-5c/setup-credentials.sh",
            "pipelines/ppe-5c/tools/run_hybrid.py",
            "pipelines/ppe-5c/tools/evaluate_pipeline.py",
            "pipelines/ppe-9c/README.md",
        )

        for relative_path in expected:
            with self.subTest(path=relative_path):
                self.assertTrue((ROOT / relative_path).is_file())

    def test_legacy_artifact_paths_are_compatibility_symlinks(self):
        expected_roots = {
            "dataset/PPE_COMMERCIAL": ROOT / "dataset/ppe-5c",
            "output/dfine_m_ppe_commercial": ROOT / "output/ppe-5c/training",
            "output/hybrid-eval": ROOT / "output/ppe-5c/evaluation",
        }

        for relative_path, target in expected_roots.items():
            path = ROOT / relative_path
            with self.subTest(path=relative_path):
                self.assertTrue(path.is_symlink())
                self.assertEqual(path.resolve(), target.resolve())


if __name__ == "__main__":
    unittest.main()

