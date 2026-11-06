import tempfile
import unittest
from pathlib import Path

import yaml

from src.ppe.registry import load_registry


class RegistryTest(unittest.TestCase):
    def test_boots_gloves_candidates_pin_versions_and_negative_mappings(self):
        registry = load_registry(Path(__file__).parents[1] / "configs/ppe/sources.yml")
        candidates = {source.id: source for source in registry.sources}

        factory = candidates["roboflow_factory_workers_v1"]
        self.assertEqual(factory.version, 1)
        self.assertEqual(factory.options["workspace"], "vishesh-mehta-9fd3l")
        self.assertEqual(factory.mappings["no_glove"], "no_gloves")
        self.assertEqual(factory.mappings["no_shoes"], "no_boots")

        apd = candidates["roboflow_apd_k3_v1"]
        self.assertEqual(apd.version, 1)
        self.assertEqual(apd.mappings["No-Gloves"], "no_gloves")
        self.assertEqual(apd.mappings["No-Shoes"], "no_boots")

        safety = candidates["roboflow_safety_ppe_v1"]
        self.assertEqual(safety.version, 1)
        self.assertEqual(safety.mappings["No_Glove"], "no_gloves")
        self.assertEqual(safety.mappings["No_Shoe"], "no_boots")

    def test_extended_ppe_candidates_are_pinned_and_mapped(self):
        registry = load_registry(Path(__file__).parents[1] / "configs/ppe/sources.yml")
        candidates = {source.id: source for source in registry.sources}

        compliance = candidates["roboflow_ppe_compliance_v2"]
        self.assertEqual(compliance.status, "pending_clearance")
        self.assertEqual(compliance.version, 2)
        self.assertEqual(compliance.sha256, "0d13990223cb4e8ded733b2958e15deddf70e8bcc8252b841ea6ab571d42c41a")
        self.assertEqual(compliance.mappings["Mask"], "mask")
        self.assertEqual(compliance.mappings["NO-Goggles"], "no_eye_protection")

        pants = candidates["roboflow_pants_v2_v1"]
        self.assertEqual(pants.status, "pending_clearance")
        self.assertEqual(pants.version, 1)
        self.assertEqual(pants.sha256, "bd0418f649ce267be0606b13ced88766028817a0c7fdd52b1b893ff94d39b7e2")
        self.assertEqual(pants.mappings["long-pants"], "long_pants")
        self.assertEqual(pants.mappings["short-pants"], "short_pants")

        construction = candidates["ultralytics_construction_ppe"]
        self.assertEqual(construction.status, "quarantined")
        self.assertEqual(construction.mappings["goggles"], "eye_protection")

    def write_registry(self, data):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "sources.yml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    def test_only_returns_commercially_approved_sources(self):
        path = self.write_registry(
            {
                "sources": [
                    {
                        "id": "allowed",
                        "status": "approved",
                        "resolver": "mendeley",
                        "dataset_id": "dataset-id",
                        "version": 1,
                        "license": "CC-BY-4.0",
                        "evidence_url": "https://example.com/license",
                        "sha256": "a" * 64,
                        "mappings": {"Helmet": "helmet"},
                    },
                    {
                        "id": "agpl",
                        "status": "quarantined",
                        "license": "AGPL-3.0",
                        "reason": "copyleft dataset",
                        "evidence_url": "https://example.com/agpl",
                    },
                ]
            }
        )

        registry = load_registry(path)

        self.assertEqual([source.id for source in registry.approved_sources()], ["allowed"])

    def test_rejects_approved_source_without_reproducible_coordinates(self):
        path = self.write_registry(
            {
                "sources": [
                    {
                        "id": "broken",
                        "status": "approved",
                        "resolver": "mendeley",
                        "dataset_id": "dataset-id",
                        "license": "CC-BY-4.0",
                        "evidence_url": "https://example.com/license",
                        "mappings": {"Helmet": "helmet"},
                    }
                ]
            }
        )

        with self.assertRaisesRegex(ValueError, "version"):
            load_registry(path)

    def test_rejects_unapproved_canonical_class(self):
        path = self.write_registry(
            {
                "sources": [
                    {
                        "id": "broken",
                        "status": "pending_hash",
                        "resolver": "mendeley",
                        "dataset_id": "dataset-id",
                        "version": 1,
                        "license": "CC-BY-4.0",
                        "evidence_url": "https://example.com/license",
                        "mappings": {"Respirator": "respirator"},
                    }
                ]
            }
        )

        with self.assertRaisesRegex(ValueError, "canonical class"):
            load_registry(path)

    def test_requires_exactly_64_lowercase_hex_characters_for_approved_hash(self):
        path = self.write_registry(
            {
                "models": [
                    {
                        "id": "pose",
                        "status": "approved",
                        "license": "Apache-2.0",
                        "evidence_url": "https://example.com/model-card",
                        "url": "https://example.com/model.task",
                        "sha256": "NOT-A-HASH",
                    }
                ]
            }
        )

        with self.assertRaisesRegex(ValueError, "sha256"):
            load_registry(path)


if __name__ == "__main__":
    unittest.main()
