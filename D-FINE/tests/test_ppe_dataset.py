import csv
import hashlib
import json
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from src.ppe.dataset import (
    HashReviewRequired,
    AnnotationRecord,
    ImageRecord,
    build_dataset,
    convert_source,
    fetch_source,
    safe_extract_zip,
)
from src.ppe.registry import SourceSpec


class BytesResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def source(**changes):
    values = {
        "id": "source",
        "status": "approved",
        "license": "CC-BY-4.0",
        "evidence_url": "https://example.com/license",
        "resolver": "roboflow",
        "version": 28,
        "sha256": hashlib.sha256(b"archive").hexdigest(),
        "mappings": {"Helmet": "helmet"},
        "options": {"workspace": "workspace", "project": "project"},
    }
    values.update(changes)
    return SourceSpec(**values)


class AcquisitionTest(unittest.TestCase):
    def test_rejects_archive_path_traversal_without_writing_outside_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../../escape.txt", "owned")

            with self.assertRaisesRegex(ValueError, "unsafe archive path"):
                safe_extract_zip(archive, root / "extract")

            self.assertFalse((root / "escape.txt").exists())

    def test_rejects_archive_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.zip"
            link = zipfile.ZipInfo("link")
            link.external_attr = 0o120777 << 16
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(link, "target")

            with self.assertRaisesRegex(ValueError, "symlink"):
                safe_extract_zip(archive, root / "extract")

    def test_roboflow_requires_api_key_before_network_access(self):
        with tempfile.TemporaryDirectory() as directory:
            called = False

            def opener(_request):
                nonlocal called
                called = True
                return BytesResponse(b"")

            with self.assertRaisesRegex(ValueError, "ROBOFLOW_API_KEY"):
                fetch_source(source(), Path(directory), {}, opener=opener)

            self.assertFalse(called)

    def test_mendeley_requires_token_before_network_access(self):
        with tempfile.TemporaryDirectory() as directory:
            called = False

            def opener(_request):
                nonlocal called
                called = True
                return BytesResponse(b"")

            with self.assertRaisesRegex(ValueError, "MENDELEY_TOKEN"):
                fetch_source(
                    source(resolver="mendeley", options={"dataset_id": "zkzghjvpn2"}),
                    Path(directory),
                    {},
                    opener=opener,
                )

            self.assertFalse(called)

    def test_mendeley_downloads_zip_from_the_pinned_version(self):
        requested = []
        responses = iter([
            b'{"files":[{"id":"archive-id","filename":"ppe.zip"}]}',
            b"archive",
        ])

        def opener(request):
            requested.append(request)
            return BytesResponse(next(responses))

        with tempfile.TemporaryDirectory() as directory:
            downloaded = fetch_source(
                source(
                    resolver="mendeley",
                    version=6,
                    options={"dataset_id": "zkzghjvpn2"},
                ),
                Path(directory),
                {"MENDELEY_TOKEN": "token"},
                opener=opener,
            )

            self.assertEqual(downloaded.read_bytes(), b"archive")
            self.assertIn("datasets/zkzghjvpn2?version=6", requested[0].full_url)
            self.assertTrue(requested[0].get_header("Authorization").startswith("Bearer "))
            self.assertTrue(requested[1].full_url.endswith("/files/archive-id"))

    def test_checksum_mismatch_does_not_publish_cache_file(self):
        responses = iter(
            [
                b'{"export":{"link":"https://download.example/archive.zip"}}',
                b"wrong archive",
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                fetch_source(
                    source(),
                    cache,
                    {"ROBOFLOW_API_KEY": "secret"},
                    opener=lambda _request: BytesResponse(next(responses)),
                )

            self.assertEqual(list(cache.rglob("*")), [])

    def test_pending_hash_preserves_download_for_review(self):
        payload = b"new archive"
        responses = iter(
            [
                b'{"export":{"link":"https://download.example/archive.zip"}}',
                payload,
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(HashReviewRequired) as raised:
                fetch_source(
                    source(status="pending_hash", sha256=None),
                    Path(directory),
                    {"ROBOFLOW_API_KEY": "secret"},
                    opener=lambda _request: BytesResponse(next(responses)),
                )

            self.assertEqual(raised.exception.sha256, hashlib.sha256(payload).hexdigest())
            self.assertEqual(raised.exception.path.read_bytes(), payload)


class ConversionTest(unittest.TestCase):
    def test_converts_coco_categories_and_preserves_source_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train").mkdir()
            Image.new("RGB", (80, 60)).save(root / "train" / "frame.jpg")
            annotation = {
                "images": [{"id": 7, "file_name": "frame.jpg", "width": 80, "height": 60}],
                "categories": [{"id": 1, "name": "Person"}, {"id": 2, "name": "Hardhat"}],
                "annotations": [{"id": 1, "image_id": 7, "category_id": 2, "bbox": [10, 5, 20, 15]}],
            }
            (root / "train" / "_annotations.coco.json").write_text(
                __import__("json").dumps(annotation), encoding="utf-8"
            )

            result = convert_source(
                source(
                    options={"workspace": "workspace", "project": "project", "format": "coco"},
                    mappings={"Person": "person", "Hardhat": "helmet"},
                ),
                root,
            )

            self.assertEqual(result.records[0].split_hint, "train")
            self.assertEqual(result.records[0].annotations[0].category, "helmet")
            self.assertEqual(result.records[0].annotations[0].bbox, (10.0, 5.0, 20.0, 15.0))

    def test_converts_declared_yolo_classes_without_inventing_negative_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images" / "train").mkdir(parents=True)
            (root / "labels" / "train").mkdir(parents=True)
            Image.new("RGB", (100, 50)).save(root / "images" / "train" / "worker.jpg")
            (root / "labels" / "train" / "worker.txt").write_text(
                "0 0.5 0.5 0.2 0.4\n", encoding="utf-8"
            )
            (root / "data.yaml").write_text(
                "names: [Helmet, Gloves]\n", encoding="utf-8"
            )

            result = convert_source(
                source(
                    resolver="mendeley",
                    options={"dataset_id": "dataset", "format": "yolo"},
                    mappings={"Helmet": "helmet", "Gloves": "gloves"},
                ),
                root,
            )

            self.assertEqual(len(result.records), 1)
            self.assertEqual(result.records[0].annotations[0].category, "helmet")
            self.assertEqual(result.records[0].annotations[0].bbox, (40.0, 15.0, 20.0, 20.0))
            self.assertNotIn(
                "no_gloves",
                [annotation.category for annotation in result.records[0].annotations],
            )

    def test_rejects_invalid_yolo_boxes_but_keeps_the_image_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images" / "train").mkdir(parents=True)
            (root / "labels" / "train").mkdir(parents=True)
            Image.new("RGB", (100, 100)).save(root / "images" / "train" / "bad.jpg")
            (root / "labels" / "train" / "bad.txt").write_text(
                "0 0.5 0.5 -0.2 0.4\n", encoding="utf-8"
            )
            (root / "data.yaml").write_text("names: [Helmet]\n", encoding="utf-8")

            result = convert_source(
                source(
                    resolver="mendeley",
                    options={"dataset_id": "dataset", "format": "yolo"},
                    mappings={"Helmet": "helmet"},
                ),
                root,
            )

            self.assertEqual(result.records[0].annotations, ())
            self.assertEqual(result.rejections[0].code, "invalid_box")


    def test_clamps_subpixel_coco_rounding_at_image_border(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train").mkdir()
            Image.new("RGB", (100, 80)).save(root / "train" / "edge.jpg")
            annotation = {
                "images": [{"id": 1, "file_name": "edge.jpg", "width": 100, "height": 80}],
                "categories": [{"id": 1, "name": "Person"}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [90, 10, 10.4, 20]}],
            }
            (root / "train" / "_annotations.coco.json").write_text(json.dumps(annotation), encoding="utf-8")

            result = convert_source(source(options={"workspace": "workspace", "project": "project", "format": "coco"}, mappings={"Person": "person"}), root)

            self.assertEqual(result.records[0].annotations[0].bbox, (90.0, 10.0, 10.0, 20.0))
            self.assertEqual(result.rejections, ())


class BuildTest(unittest.TestCase):
    def test_custom_taxonomy_controls_output_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "worker.jpg"
            Image.new("RGB", (100, 100), "white").save(image)
            record = ImageRecord(
                "source/worker", "source", "worker.jpg", image, 100, 100,
                "source/origin", "train",
                (
                    AnnotationRecord("person", (5, 5, 90, 90), "Person"),
                    AnnotationRecord("helmet", (30, 5, 30, 20), "Helmet"),
                    AnnotationRecord("mask", (35, 25, 20, 15), "Mask"),
                    AnnotationRecord("no_mask", (35, 25, 20, 15), "NO-Mask"),
                    AnnotationRecord("long_pants", (25, 55, 50, 35), "long-pants"),
                ),
            )

            report = build_dataset(
                [record], root / "output", seed=7,
                detector_classes=("person", "helmet", "mask", "no_mask"),
                state_classes=("helmet", "mask", "no_mask"),
                context_classes=(),
            )
            instances = json.loads((report.path / "annotations" / "instances_train.json").read_text())
            states = json.loads((report.path / "annotations" / "states_train.json").read_text())

            self.assertEqual([item["name"] for item in instances["categories"]], ["person", "helmet", "mask", "no_mask"])
            self.assertEqual([item["name"] for item in states["categories"]], ["helmet", "mask", "no_mask"])
            self.assertEqual({item["category_id"] for item in instances["annotations"]}, {0, 1, 2, 3})
            self.assertNotIn("long_pants", json.dumps(instances) + json.dumps(states))

    def test_rare_ppe_gets_context_crop_in_the_parent_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "worker.jpg"
            Image.new("RGB", (200, 200), "white").save(image)
            record = ImageRecord(
                "source/worker", "source", "worker.jpg", image, 200, 200,
                "source/origin", "train",
                (AnnotationRecord("gloves", (90.0, 80.0, 10.0, 10.0), "Gloves"),),
            )

            report = build_dataset([record], root / "output", seed=2)
            with (report.path / "manifest.csv").open(newline="", encoding="utf-8") as handle:
                manifest = list(csv.DictReader(handle))
            parent = next(row for row in manifest if row["record_id"] == "source/worker")
            crop = next(row for row in manifest if row["image_kind"] == "gloves_context")

            self.assertEqual(crop["parent_id"], parent["record_id"])
            self.assertEqual(crop["split"], parent["split"])
            self.assertEqual(report.audit["context_crops"], 1)
            annotations = json.loads((report.path / "annotations" / "instances_train.json").read_text())
            boxes = [item["bbox"] for item in annotations["annotations"] if item["image_id"] == next(image["id"] for image in annotations["images"] if image["file_name"] == crop["output_image"])]
            self.assertEqual(boxes, [[27.0, 27.0, 10.0, 10.0]])

    def test_person_crop_is_materialized_in_detector_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "person.jpg"
            Image.new("RGB", (100, 100), "white").save(image)
            record = ImageRecord(
                "source/person", "source", "person.jpg", image, 100, 100,
                "source/origin", "train",
                (
                    AnnotationRecord("person", (10.0, 5.0, 80.0, 90.0), "Person"),
                    AnnotationRecord("helmet", (35.0, 10.0, 30.0, 20.0), "Helmet"),
                ),
            )

            report = build_dataset([record], root / "output", seed=3)
            annotations = json.loads(
                (report.path / "annotations" / "instances_train.json").read_text()
            )

            self.assertEqual(len(annotations["images"]), 2)
            crop = next(image for image in annotations["images"] if "person-0" in image["file_name"])
            crop_boxes = [
                annotation["bbox"]
                for annotation in annotations["annotations"]
                if annotation["image_id"] == crop["id"]
            ]
            self.assertIn([33.0, 10.0, 30.0, 20.0], crop_boxes)

    def test_groups_duplicates_before_split_and_keeps_crop_with_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jpg"
            second = root / "second.jpg"
            Image.new("RGB", (100, 100), "red").save(first)
            second.write_bytes(first.read_bytes())
            person = AnnotationRecord("person", (10.0, 5.0, 80.0, 90.0), "Person")
            helmet = AnnotationRecord("helmet", (35.0, 10.0, 30.0, 20.0), "Helmet")
            records = [
                ImageRecord(
                    "a/first", "a", "first.jpg", first, 100, 100, "a/origin-1", "train", (person, helmet)
                ),
                ImageRecord(
                    "b/second", "b", "second.jpg", second, 100, 100, "b/origin-2", "valid", (person,)
                ),
            ]

            report = build_dataset(
                records, root / "output", seed=7,
                detector_classes=("person", "helmet"),
                state_classes=("helmet",),
                context_classes=(),
            )
            with (report.path / "manifest.csv").open(newline="", encoding="utf-8") as handle:
                manifest = list(csv.DictReader(handle))

            full_frames = [row for row in manifest if row["image_kind"] == "full_frame"]
            self.assertEqual([row["record_id"] for row in full_frames], ["a/first"])
            self.assertEqual(report.audit["deduplicated_full_frames"], 1)
            parent = next(row for row in manifest if row["record_id"] == "a/first")
            crop = next(row for row in manifest if row["parent_id"] == "a/first")
            self.assertEqual(crop["split"], parent["split"])
            self.assertEqual(report.audit["cross_split_duplicate_groups"], 0)

    def test_same_inputs_and_seed_reuse_the_same_immutable_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.jpg"
            Image.new("RGB", (32, 32), "blue").save(image)
            record = ImageRecord(
                "source/image", "source", "image.jpg", image, 32, 32,
                "source/origin", "train", (),
            )

            first = build_dataset([record], root / "output", seed=11)
            manifest = (first.path / "manifest.csv").read_bytes()
            second = build_dataset([record], root / "output", seed=11)

            self.assertEqual(first.build_id, second.build_id)
            self.assertEqual((second.path / "manifest.csv").read_bytes(), manifest)
            self.assertEqual(json.loads((second.path / "audit.json").read_text())["images"], 1)


    def test_rejections_and_license_provenance_are_published_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.jpg"
            Image.new("RGB", (32, 32), "green").save(image)
            record = ImageRecord(
                "source/image", "source", "image.jpg", image, 32, 32,
                "source/origin", "train", (AnnotationRecord("helmet", (1, 2, 3, 4), "Hat"),),
            )
            rejections = [{"source_id": "source", "source_image_id": "image.jpg", "code": "unmapped_class", "detail": "Other"}]
            licenses = [{"source_id": "source", "license": "CC-BY-4.0", "evidence_url": "https://example.com/license", "version": 1, "sha256": "a" * 64}]

            report = build_dataset([record], root / "output", seed=13, rejections=rejections, licenses=licenses)

            self.assertTrue((report.path / "rejected.csv").exists())
            self.assertTrue((report.path / "licenses.csv").exists())
            self.assertIn("CC-BY-4.0", (report.path / "ATTRIBUTION.md").read_text())
            self.assertEqual(report.audit["classes"], {"helmet": 1})
            self.assertEqual(report.audit["sources"], {"source": 1})
            self.assertEqual(report.audit["licenses"], {"CC-BY-4.0": 1})
            self.assertEqual(report.audit["rejected_reasons"], {"unmapped_class": 1})
            self.assertEqual(report.audit["box_size_buckets"], {"small": 1})


if __name__ == "__main__":
    unittest.main()
