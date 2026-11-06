import unittest
from pathlib import Path

from src.core import YAMLConfig


class PPEConfigTest(unittest.TestCase):
    def test_resolves_five_class_960px_finetuning(self):
        root = Path(__file__).parents[2]
        config = YAMLConfig(
            str(root / "pipelines" / "ppe-5c" / "configs" / "dfine_m_960.yml")
        ).yaml_cfg

        self.assertEqual(config["num_classes"], 5)
        self.assertFalse(config["remap_mscoco_category"])
        self.assertEqual(config["train_dataloader"]["total_batch_size"], 2)
        self.assertEqual(config["val_dataloader"]["total_batch_size"], 2)
        self.assertEqual(config["train_dataloader"]["collate_fn"]["base_size"], 960)
        self.assertEqual(config["eval_spatial_size"], [960, 960])
        self.assertEqual(
            next(
                operation["size"]
                for operation in config["train_dataloader"]["dataset"]["transforms"]["ops"]
                if operation["type"] == "Resize"
            ),
            [960, 960],
        )
        self.assertEqual(config["optimizer"]["lr"], 0.000015625)
        self.assertEqual(config["optimizer"]["params"][0]["lr"], 0.0000015625)
        self.assertFalse(config["HGNetv2"]["pretrained"])
        self.assertNotEqual(config["output_dir"], "./output/dfine_hgnetv2_m_custom")

    def test_resolves_candidate_training_configs(self):
        root = Path(__file__).parents[2]
        cases = {
            "ppe-13c": (13, root / "dataset/candidates/ppe-13c/current"),
            "pants-2c": (2, root / "dataset/candidates/pants-2c/current"),
        }
        for name, (num_classes, dataset) in cases.items():
            with self.subTest(candidate=name):
                config = YAMLConfig(str(root / "pipelines/candidates" / name / "configs/dfine_m_960.yml")).yaml_cfg
                self.assertEqual(config["num_classes"], num_classes)
                self.assertEqual(config["eval_spatial_size"], [960, 960])
                self.assertEqual(config["train_dataloader"]["total_batch_size"], 2)
                self.assertEqual(config["train_dataloader"]["dataset"]["img_folder"], str(dataset))
                self.assertEqual(config["train_dataloader"]["dataset"]["ann_file"], str(dataset / "annotations/instances_train.json"))
                self.assertEqual(config["val_dataloader"]["dataset"]["ann_file"], str(dataset / "annotations/instances_val.json"))
                self.assertEqual(config["output_dir"], str(root / "output/candidates" / name / "training"))


if __name__ == "__main__":
    unittest.main()
