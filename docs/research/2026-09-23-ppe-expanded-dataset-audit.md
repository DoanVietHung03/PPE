# Expanded PPE dataset audit — 2026-09-23

Sources were downloaded for internal audit only. No source was merged into the
canonical `dataset/ppe-5c` build.

## Archives

| Source | Archive SHA-256 | Exported images | Annotations | Decision |
|---|---|---:|---:|---|
| A — PPE Detection and Compliance v2 | `0d13990223cb4e8ded733b2958e15deddf70e8bcc8252b841ea6ab571d42c41a` | 9,663 | 39,510 | Candidate for expanded PPE taxonomy |
| B — Pants_v2 v1 | `bd0418f649ce267be0606b13ced88766028817a0c7fdd52b1b893ff94d39b7e2` | 796 | 899 | Candidate for a separate pants branch |
| C — Ultralytics Construction-PPE | `bef8dcb599aa4e9d9f5e602cb6fa7143d3c84d7f6a0ff40463d7f2a4c2632ccc` | 1,416 | 11,614 | Quarantined: AGPL-3.0 and mostly already present |

Roboflow advertises 10,080 images for A, while its COCO export contains 9,663
image records. The exported archive is the reproducible unit recorded above.

## Class coverage

- A contains all 13 PPE/person classes. Its rarest class is `NO-Boots` with
  776 boxes; the largest are `Vest` 4,969 and `Person` 4,930.
- B contains 616 `long-pants` and 283 `short-pants` boxes.
- C contains 11 classes including the unmapped `none`; `no_boots` has only 115
  boxes.

## Duplicate audit

Across A, B, C, and the current 31,845-image build:

- Exact SHA-256: A has 62 internal duplicate files; current has 445; C has
  1,276 exact matches already in the current dataset.
- Same-size exact dHash: A has 244 internal duplicate files, B has 9, C has 15,
  and current has 1,490. Cross-source groups: A/current 123, A/C 13, and
  C/current 1,296.
- Source-provided split leakage by same-size exact dHash: A 56 groups, B 3,
  C 4. Any future build must regroup before splitting.

## Integration decision

Do not concatenate the exports. A lacks pants annotations and B lacks PPE annotations, so a conventional single detector would learn visible unlabeled objects as background. Keep the production five-class dataset unchanged. Build an expanded candidate only after choosing either exhaustive relabeling or separate PPE and pants detector branches. Source C requires license clearance; the current five-class audit already records one AGPL-3.0 source and 1,276 C full frames, which should be reviewed before commercial deployment.

## Generated candidates

Both candidates were rebuilt from their pinned archive with seed `20260923`. The builder regrouped source variants, removed near-duplicates before splitting, and reported zero cross-split duplicate groups.

| Candidate | Build ID | Full frames after dedup | Generated images | Train / val / test images |
|---|---|---:|---:|---:|
| PPE 13-class | `f9f77e2427b162b2` | 8,794 | 17,174 | 13,764 / 1,713 / 1,697 |
| Pants 2-class | `f84670822dfee0cf` | 758 | 758 | 607 / 75 / 76 |

The PPE build removed 869 duplicate full frames and generated 4,596 person crops plus 3,784 gloves/boots context crops. The pants build removed 38 duplicate full frames. Neither build rejected an annotation.

Every declared category has nonzero annotations in train, validation and test. The rarest PPE category remains `no_boots` with 865 / 91 / 107 generated annotations. Pants coverage is 474 / 55 / 64 `long_pants` and 212 / 26 / 30 `short_pants`.

Each candidate contains exactly one CC-BY-4.0 provenance row and neither manifest contains `ultralytics_construction_ppe`. Production remained unchanged: `dataset/ppe-5c/CURRENT` is still `6a4dd4f1292433d7`, and its audit plus both configuration file hashes match the pre-build values.
