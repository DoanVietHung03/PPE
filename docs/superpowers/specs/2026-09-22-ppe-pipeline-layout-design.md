# PPE Pipeline Layout Design

**Date:** 2026-09-22

## Goal

Separate the working five-class hybrid pipeline from the future nine-class
direct detector, and give the five-class pipeline one obvious home.

## Taxonomies

- `ppe-5c`: `person`, `helmet`, `vest`, `gloves`, `boots`. Negative PPE states
  are inferred by the pose-gated policy.
- `ppe-9c`: the five classes above plus `no_helmet`, `no_vest`, `no_gloves`,
  and `no_boots` as detector outputs.
- The legacy eight-class experiment has no production namespace. Its invalid
  `NaN.pth` checkpoint remains archived and is never treated as a deployable
  model.

## Canonical Layout

```text
pipelines/
├── ppe-5c/
│   ├── README.md
│   ├── configs/
│   ├── deploy/
│   └── tools/
└── ppe-9c/
    └── README.md

dataset/ppe-5c/
models/ppe-5c/
output/ppe-5c/
```

`D-FINE/` remains the shared upstream framework. Framework internals and shared
training modules are not copied into either pipeline directory.

## Five-Class Contents

- `configs/`: the canonical 960px training/dataset configuration.
- `deploy/`: the standalone ONNX + MediaPipe application and its local `ppe`
  package.
- `tools/`: hybrid PyTorch inference and state evaluation entry points.
- `README.md`: taxonomy, canonical paths, training, export, validation, and
  deployment commands.

Tests remain under `D-FINE/tests` because they exercise the shared framework as
well as pipeline entry points.

## Artifact Migration

- Rename the commercial dataset root to `dataset/ppe-5c`.
- Place deployable ONNX and pose assets under `models/ppe-5c` while keeping the
  approved upstream COCO checkpoint under `models/approved`.
- Rename training and evaluation outputs under `output/ppe-5c`.
- Update every repository-owned config, script, test, and document that invokes
  a canonical path.
