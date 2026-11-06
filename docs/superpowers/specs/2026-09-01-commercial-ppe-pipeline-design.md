# Commercial PPE Pipeline Design

**Date:** 2026-09-01

## Goal

Build a reproducible, fully automated PPE training and inference pipeline for commercial use. It must support CCTV full frames and pre-cropped people, preserve the eight public PPE states (`helmet`, `vest`, `gloves`, `boots`, `no_helmet`, `no_vest`, `no_gloves`, `no_boots`), and avoid treating an occluded or out-of-frame body part as missing PPE.

## Constraints

- No manual annotation or review is available in the data pipeline.
- Accuracy has priority over throughput.
- Every dataset, model asset, and derived image must have recorded provenance, version, license, and checksum.
- AGPL, non-commercial, unclear-provenance, and unverified assets are quarantined from the commercial build by default.
- Existing user changes in `D-FINE/src/solver/det_engine.py` and `D-FINE/tests/` must be preserved.
- The existing `dataset/PPE_DFINE` and `output/dfine_m_ppe` are immutable baselines, not overwritten.

## Architecture

The production pipeline is hybrid rather than a direct eight-class detector:

1. D-FINE-M detects `person`, `helmet`, `vest`, `gloves`, and `boots`.
2. Each detected person is cropped with bounded padding.
3. BlazePose Heavy estimates the visibility of head, torso, wrists/hands, ankles, heels, and feet for that one crop.
4. PPE detections are assigned to a person by containment and overlap with the corresponding body region.
5. A deterministic policy emits the eight public PPE states. A `no_*` state is emitted only when its body region is visible with sufficient keypoint confidence and no matching positive PPE is present. Otherwise that state is `unknown`.

The public result therefore retains all eight requested labels while separating observable absence from missing evidence.

## Why This Architecture

A direct eight-class detector cannot learn a commercially usable `no_boots` class from the available cleared data: the legacy dataset contains only 90 boxes in total, and the main matching source is declared AGPL-3.0. Automatically interpreting a missing positive annotation as `no_*` would create false labels. The visibility gate closes that semantic gap without inventing negatives.

BlazePose Heavy is run once per D-FINE person crop. Its exact model asset has an Apache-2.0 model card, and per-person cropping avoids its single-person limitation. Its published validation does not cover CCTV surveillance, so it remains subject to the domain acceptance tests below. RTMPose is the technical fallback only after the exact checkpoint or its training lineage receives written commercial clearance.

## Dataset Sources and Commercial Gate

The first commercial build imports only sources that pass an allowlist with immutable evidence:

- Mendeley `Dataset of Personal Protective Equipment (PPE), v6`, CC BY 4.0, for `helmet`, `vest`, `no_helmet`, and `no_vest`.
- Roboflow `Construction Site Safety`, CC BY 4.0, for `person`, `helmet`, `vest`, `gloves`, `no_helmet`, and `no_vest`.
- Existing `construction_local`, `worker_safety_2`, and `worker_safety_25` only when their original URL/version and license evidence are added to the source registry and their checksums match.

The following are excluded by default:

- `ultralytics_construction_ppe`: declared AGPL-3.0 and lacks independently documented per-image provenance.
- SH17: CC BY-NC-SA 4.0.
- SHWD and Pictor-PPE: image provenance or dataset license is insufficient for commercial approval.
- Roboflow community datasets whose card states CC BY but does not document image provenance.
- Open Images files whose individual image license and attribution cannot be verified at ingest time.

An excluded source can enter only by changing its registry status alongside stored written clearance. No command-line override bypasses this gate.

## Canonical Data Model

Every accepted image record contains:

- canonical image path and SHA-256;
- perceptual hash for near-duplicate grouping;
- source dataset, source version, source image ID, source URL, author/attribution, license identifier, and captured license evidence path;
- origin group and media group when known;
- image kind: `full_frame` or `person_crop`;
- parent image ID and crop box for derived crops;
- split and mapped annotations;
- every mapping, crop, resize, and filtering transformation applied.

Annotations use canonical COCO boxes. Source labels are mapped only through an explicit source registry. `shoes` is not mapped to `boots` unless the source ontology explicitly means safety boots. Absence of a source annotation never creates a negative PPE annotation.

## Build Flow

1. Resolve only allowlisted, version-pinned sources.
2. Download into a cache and verify the declared archive checksum before extraction.
3. Convert source annotations to the canonical representation.
4. Reject invalid boxes, unknown mappings, incomplete license records, and unsafe paths.
5. Compute exact and perceptual duplicate groups across all sources.
6. Assign train, validation, and test by origin/media group, stratified by source and available class labels.
7. Generate deterministic person crops after split assignment; parent and all derivatives inherit the same split and license.
8. Materialize COCO annotations for the five detector classes and evaluation records for explicit `no_*` labels.
9. Write manifest, rejected ledger, attribution notice, license ledger, and audit report atomically into a versioned output directory.

The build never mutates a previous dataset version. Re-running with the same registry and inputs produces the same manifest and split.

## Full-Frame and Crop Balance

Training samples both image kinds without duplicating rare origins. The initial target is equal probability for full-frame and person-crop batches. Source balance is applied at origin-group level, not by repeatedly copying the same rare images. Validation and test preserve their natural distribution and report each image kind separately.

Person crops include bounded context around the person and never extend outside the image. Crops with insufficient resolution for the relevant body regions remain useful for positive detection but their unavailable PPE states evaluate as `unknown`, not negative.

## Model and Training

D-FINE-M remains the detector because increasing model size has not been justified. The class head changes to five classes. Training begins from a model asset only after its exact URL, SHA-256, model card/license, and training-data caveat are stored in the model registry. Objects365-derived checkpoints are excluded by default.

The initial input size is 800 pixels. Resolution 960 is a single controlled follow-up only if the clean validation report shows small-person or glove/boot localization remains the limiting bucket.

The optimizer learning rates are scaled from the upstream batch-32 recipe to the actual total batch size of four. Existing augmentations are retained unless the clean baseline demonstrates a specific failure. The run keeps deterministic seeds, records the resolved configuration, evaluates every epoch, and selects the checkpoint by the validation score described below rather than by final epoch.

## PPE State Policy

For each person, the policy processes four body regions:

- head for helmet;
- upper torso for vest;
- left and right hands for gloves;
- left and right feet for boots.

A region is visible only when its required keypoints exceed the configured confidence and lie inside the uncropped image area. Positive PPE takes precedence when a matching detection passes its class threshold and spatial association rule. `no_*` is emitted only for a visible region without matched positive PPE. Any low-confidence, occluded, truncated, or contradictory case emits `unknown`.

Thresholds are calibration parameters stored in the resolved experiment configuration. They are selected on validation data and frozen before test evaluation.

## Evaluation and Acceptance

The immutable legacy model and new pipeline are evaluated on the same newly built, commercially allowed test set.

Detector reporting includes COCO AP, AP50, AP75, and AP by object size for each of the five detector classes, split by full-frame and person-crop data.

End-to-end reporting treats each person/PPE state as `present`, `absent`, or `unknown` and includes:

- precision, recall, and F1 for each of the eight public labels;
- macro-F1 and no-PPE macro recall;
- false alarms per evaluated person;
- `unknown` coverage per PPE type;
- results by source, image kind, person size, occlusion, and day/night bucket when metadata exists.

The new pipeline is accepted only when it improves macro-F1 and no-PPE macro recall over the clean baseline, does not materially regress any supported label, and does not obtain the gain by excessive `unknown` output. Exact numeric gates are set from the first clean baseline rather than fabricated from the contaminated legacy split.

`no_gloves` and `no_boots` test metrics are reported only when the clean test set has enough explicit ground-truth cases for a meaningful result. Until then they are reported as insufficient-data, while policy output is still exercised with deterministic unit and synthetic geometry cases.

## Validation and Failure Handling

The smallest required checks are:

- source-registry schema and license allowlist tests;
- malicious archive path and checksum rejection tests;
- class mapping and invalid-box tests;
- exact/near-duplicate grouping and group-split leakage tests;
- deterministic crop and rebuild tests;
- policy tests for positive, visible-negative, occluded, truncated, low-confidence, and contradictory states;
- a small end-to-end fixture that builds a dataset and evaluates known predictions;
- PyTorch-to-export parity checks for every deployed model artifact.

A failed source, license, checksum, or validation gate aborts the new build without modifying the last successful version. Rejected records remain in a machine-readable ledger with a stable reason code.

## Deliverables

- A versioned source/model registry with license evidence and hashes.
- A reproducible dataset builder for approved sources, canonical conversion, deduplication, group splitting, crop generation, and audit output.
- A five-class D-FINE-M configuration with corrected pretrained handling, batch-scaled learning rate, and 800-pixel training.
- A per-person pose/visibility and PPE-state policy that returns all eight public labels plus `unknown` internally.
- Clean baseline and comparison commands producing per-class and end-to-end reports.
- Updated operating documentation describing source approval, rebuild, train, evaluate, and export commands.

## Explicitly Deferred

- Larger D-FINE variants: add only if the clean error analysis shows capacity, not data, is the bottleneck.
- 960-pixel training: add only if the 800-pixel small-object bucket remains limiting.
- RTMPose: add only after exact checkpoint/training lineage receives written commercial clearance.
- Synthetic or inpainted `no_*` data: add only as a controlled experiment after the clean real-data baseline.
- Online tracking and temporal voting: add only after frame-level accuracy passes and video flicker is measured.

## Research References

- `docs/research/ppe-commercial-datasets.md`
- `docs/research/ppe-pose-models-commercial.md`

