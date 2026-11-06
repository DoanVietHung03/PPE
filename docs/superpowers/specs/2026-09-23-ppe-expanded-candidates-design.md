# PPE expanded dataset candidates

**Date:** 2026-09-23

## Goal

Create two reproducible training candidates from the audited archives without
changing the working five-class production dataset:

- `dataset/candidates/ppe-13c`: person plus six positive/negative PPE pairs
  from PPE Detection and Compliance v2.
- `dataset/candidates/pants-2c`: long pants and short pants from Pants_v2 v1.

The candidates remain separate because each source leaves the other source's
objects unlabeled. They must not be concatenated into a conventional 15-class
detector dataset.

## Taxonomies

`ppe-13c`, in model order:

1. `person`
2. `helmet`
3. `no_helmet`
4. `vest`
5. `no_vest`
6. `gloves`
7. `no_gloves`
8. `boots`
9. `no_boots`
10. `mask`
11. `no_mask`
12. `eye_protection`
13. `no_eye_protection`

`pants-2c`, in model order:

1. `long_pants`
2. `short_pants`

`unknown` remains a policy state, not a detector class.

## Build flow

Extend the existing dataset builder with an explicit taxonomy selected by the
candidate command. Reuse its safe ZIP extraction, annotation validation,
provenance manifest, exact/perceptual deduplication, origin grouping and
deterministic split assignment. Do not add a second builder implementation.

Each candidate writes immutable content-addressed builds plus `CURRENT` and
`current`, matching the existing dataset convention. The source archive hash,
license URL, class counts, rejection counts and duplicate counts are retained
in the audit artifacts.

Source-provided train/validation/test splits are hints only. All records are
regrouped and split again so exact images, matching dHash images and Roboflow
variants cannot cross candidate splits.

## Training boundary

Add candidate D-FINE configs only after the generated COCO files verify. The
13-class config may initialize from the existing COCO-pretrained D-FINE model;
the pants config is an independent two-class experiment. Neither candidate
replaces `pipelines/ppe-5c`, its model, or its deployment defaults.

No combined inference policy is added in this change. Integration happens only
after each branch has independent validation metrics and real-camera tests.

## Verification

