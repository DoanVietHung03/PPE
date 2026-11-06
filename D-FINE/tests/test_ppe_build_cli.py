import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from tools.ppe import build_dataset as cli
from tools.ppe.build_dataset import _publish_current


class CandidateBuildTest(unittest.TestCase):
    def test_build_candidate_selects_one_source_and_taxonomy(self):
        sources = (
            SimpleNamespace(id="roboflow_ppe_compliance_v2", status="pending_clearance", sha256="a" * 64, license="CC-BY-4.0", evidence_url="https://example.com/a", version=2),
            SimpleNamespace(id="roboflow_pants_v2_v1", status="pending_clearance", sha256="b" * 64, license="CC-BY-4.0", evidence_url="https://example.com/b", version=1),
        )
        cases = {
            "ppe-13c": (sources[0], ("person", "helmet", "no_helmet", "vest", "no_vest", "gloves", "no_gloves", "boots", "no_boots", "mask", "no_mask", "eye_protection", "no_eye_protection")),
            "pants-2c": (sources[1], ("long_pants", "short_pants")),
        }
        for candidate, (expected_source, expected_classes) in cases.items():
            with self.subTest(candidate=candidate), tempfile.TemporaryDirectory() as directory, \
                 patch.object(cli, "load_registry", return_value=SimpleNamespace(sources=sources)), \
                 patch.object(cli, "fetch_source", return_value=Path(directory) / "archive.zip"), \
                 patch.object(cli, "_extracted_root", return_value=Path(directory)), \
                 patch.object(cli, "convert_source", return_value=SimpleNamespace(records=("record",), rejections=())), \
                 patch.object(cli, "build_dataset", return_value=SimpleNamespace(build_id="build-id", path=Path(directory))) as build, \
                 patch.object(cli, "_publish_current"):
                cli.main(["build-candidate", "--candidate", candidate, "--registry", "registry.yml", "--cache", directory, "--output", str(Path(directory) / candidate)])
                self.assertIs(cli.convert_source.call_args.args[0], expected_source)
                self.assertEqual(build.call_args.kwargs["detector_classes"], expected_classes)

    def test_candidate_rejects_output_for_another_pipeline(self):
        source = SimpleNamespace(id="roboflow_ppe_compliance_v2", status="pending_clearance", sha256="a" * 64, license="CC-BY-4.0", evidence_url="https://example.com/a", version=2)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(cli, "load_registry", return_value=SimpleNamespace(sources=(source,))), \
             patch.object(cli, "fetch_source", return_value=Path(directory) / "archive.zip"), \
             patch.object(cli, "_extracted_root", return_value=Path(directory)), \
             patch.object(cli, "convert_source", return_value=SimpleNamespace(records=("record",), rejections=())), \
             patch.object(cli, "build_dataset", return_value=SimpleNamespace(build_id="build-id", path=Path(directory))), \
             patch.object(cli, "_publish_current"):
            with self.assertRaisesRegex(ValueError, "output directory must end with ppe-13c"):
                cli.main(["build-candidate", "--candidate", "ppe-13c", "--registry", "registry.yml", "--cache", directory, "--output", str(Path(directory) / "ppe-5c")])

    def test_candidate_fetch_failure_does_not_publish_current(self):
        source = SimpleNamespace(id="roboflow_ppe_compliance_v2", status="pending_clearance", sha256="a" * 64)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(cli, "load_registry", return_value=SimpleNamespace(sources=(source,))), \
             patch.object(cli, "fetch_source", side_effect=ValueError("cached checksum mismatch")):
            output = Path(directory) / "ppe-13c"
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                cli.main(["build-candidate", "--candidate", "ppe-13c", "--registry", "registry.yml", "--cache", directory, "--output", str(output)])
            self.assertFalse((output / "CURRENT").exists())
            self.assertFalse((output / "current").exists())


class BuildPointerTest(unittest.TestCase):
    def test_publishes_atomic_current_symlink_for_training_config(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "build-a").mkdir()
            (output / "build-b").mkdir()

            _publish_current(output, "build-a")
            _publish_current(output, "build-b")

            self.assertEqual((output / "CURRENT").read_text().strip(), "build-b")
            self.assertEqual((output / "current").resolve(), output / "build-b")


if __name__ == "__main__":
    unittest.main()
