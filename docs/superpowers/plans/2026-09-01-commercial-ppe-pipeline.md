# Commercial PPE Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, train, and evaluate a commercially gated hybrid PPE pipeline for CCTV frames and person crops while preserving the eight requested PPE labels.

**Architecture:** D-FINE-M detects people and four positive PPE classes. A per-person BlazePose visibility pass and deterministic policy emit reliable `no_*` states or `unknown`; a versioned dataset builder admits only allowlisted sources, deduplicates by origin before splitting, and derives crops after splitting.

**Tech Stack:** Python 3.12, PyTorch/D-FINE, torchvision, PyYAML, Pillow, faster-coco-eval, MediaPipe Tasks, ONNX Runtime, standard-library `unittest`.

**Spec:** `../docs/superpowers/specs/2026-09-01-commercial-ppe-pipeline-design.md` from the `D-FINE` repository root.

## Global Constraints

- Support CCTV full frames and pre-cropped people.
- Preserve public labels `helmet`, `vest`, `gloves`, `boots`, `no_helmet`, `no_vest`, `no_gloves`, and `no_boots`.
- Emit `no_*` only when the corresponding body region is visibly evaluable; otherwise emit `unknown`.
- Do not manually annotate data or infer negative labels from missing positive annotations.
- Quarantine AGPL, non-commercial, unclear-provenance, and unverified sources by default.
- Record exact source/model URL, version, license evidence, and SHA-256.
- Do not overwrite `dataset/PPE_DFINE` or `output/dfine_m_ppe`.
- Preserve existing user changes in `D-FINE/src/solver/det_engine.py` and `D-FINE/tests/test_det_engine.py`.
- Prefer accuracy over throughput; start at 800-pixel detector input and defer 960 until measured.
- The workspace root is not a Git repository. Commit only files inside the `D-FINE` repository and never stage the user's existing dirty files.

---

## File Structure

- `D-FINE/configs/ppe/sources.yml`: immutable source/model registry and class mappings.
- `D-FINE/src/ppe/registry.py`: registry parsing and commercial allowlist validation.
- `D-FINE/src/ppe/dataset.py`: safe acquisition, canonical conversion, deduplication, grouped splitting, crop generation, and atomic materialization.
- `D-FINE/src/ppe/policy.py`: body-region visibility, PPE/person association, and eight-label state policy.
- `D-FINE/src/ppe/__init__.py`: the small supported PPE module surface.
- `D-FINE/tools/ppe/build_dataset.py`: dataset build CLI.
- `D-FINE/tools/ppe/evaluate_pipeline.py`: detector and end-to-end evaluation CLI.
- `D-FINE/tools/ppe/run_hybrid.py`: image/video inference CLI.
- `D-FINE/configs/dfine/custom/ppe_detection.yml`: five-class commercial dataset config.
- `D-FINE/configs/dfine/custom/dfine_hgnetv2_m_ppe.yml`: batch-scaled 800-pixel training config.
- `D-FINE/tests/test_ppe_registry.py`: source/model gate regression tests.
- `D-FINE/tests/test_ppe_dataset.py`: conversion, deduplication, split, crop, and reproducibility tests.
- `D-FINE/tests/test_ppe_policy.py`: visibility and positive/negative/unknown policy tests.
- `D-FINE/tests/test_ppe_evaluation.py`: metric aggregation tests.
- `D-FINE/requirements-ppe.txt`: only the additional runtime packages needed by the approved hybrid pipeline.
- Generated `dataset/PPE_COMMERCIAL/<build-id>/`: immutable images, COCO files, manifest, rejected ledger, attribution, licenses, and audit.
- Generated `output/dfine_m_ppe_commercial/<run-id>/`: resolved config, logs, checkpoints, and evaluation reports.

---

### Task 1: Commercial Source and Model Registry

**Files:**
- Create: `D-FINE/configs/ppe/sources.yml`
- Create: `D-FINE/src/ppe/__init__.py`
- Create: `D-FINE/src/ppe/registry.py`
- Create: `D-FINE/tests/test_ppe_registry.py`

**Interfaces:**
- Produces: `load_registry(path: Path) -> Registry`
- Produces: `Registry.approved_sources() -> tuple[SourceSpec, ...]`
- Produces: `Registry.require_approved_model(name: str) -> ModelSpec`
- Consumes: no project-specific implementation beyond PyYAML.

- [ ] **Step 1: Write the failing registry tests**

```python
def test_commercial_registry_rejects_agpl_and_missing_evidence(tmp_path):
    path = write_registry(tmp_path, sources=[
        source("allowed", "CC-BY-4.0", "approved", evidence="license.txt"),
        source("agpl", "AGPL-3.0", "quarantined", evidence="agpl.txt"),
    ])
    registry = load_registry(path)
    assert [source.name for source in registry.approved_sources()] == ["allowed"]

def test_registry_requires_version_checksum_and_class_mapping(self):
    with tempfile.TemporaryDirectory() as directory:
        path = write_registry(
            Path(directory),
            sources=[source("broken", "CC-BY-4.0", "approved")],
        )
        with self.assertRaisesRegex(ValueError, "version"):
            load_registry(path)
```

The test module imports `tempfile`, `Path`, and `unittest`; it must not add pytest as a dependency.

- [ ] **Step 2: Run tests and verify red**

Run: `cd D-FINE && ../venv/bin/python -m unittest tests.test_ppe_registry -v`

Expected: import failure for `src.ppe.registry`.

- [ ] **Step 3: Implement the minimal typed registry**

Use frozen dataclasses for `SourceSpec`, `ModelSpec`, and `Registry`. Validate required fields, canonical class names, unique IDs, approved-license allowlist, evidence paths, resolver-specific coordinates, and 64-character lowercase SHA-256 when a fixed archive is declared.

Seed `sources.yml` with:

```yaml
sources:
  - id: mendeley_ppe_v6
    status: approved
    resolver: mendeley
    dataset_id: zkzghjvpn2
    version: 6
    license: CC-BY-4.0
    evidence_url: https://data.mendeley.com/datasets/zkzghjvpn2/6
    mappings: {Helmet: helmet, Vest: vest, NoHelmet: no_helmet, NoVest: no_vest}
  - id: roboflow_construction_site_safety_v28
    status: approved
    resolver: roboflow
    workspace: roboflow-universe-projects
    project: construction-site-safety
    version: 28
    license: CC-BY-4.0
    evidence_url: https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/28
    mappings: {Person: person, Hardhat: helmet, Safety-Vest: vest, Gloves: gloves, NO-Hardhat: no_helmet, NO-Safety-Vest: no_vest}
  - id: ultralytics_construction_ppe
    status: quarantined
    reason: AGPL-3.0 and incomplete per-image provenance
    license: AGPL-3.0
    evidence_url: https://docs.ultralytics.com/datasets/detect/construction-ppe/
```

Register BlazePose Heavy by exact model-card URL and expected asset hash only after the download resolver obtains and prints the hash for explicit review. Keep RTMPose and Objects365 checkpoints quarantined.

- [ ] **Step 4: Run registry tests and full existing tests**

Run: `cd D-FINE && ../venv/bin/python -m unittest tests.test_ppe_registry tests.test_det_engine -v`

Expected: all tests pass.

- [ ] **Step 5: Commit only Task 1 files**

```bash
cd D-FINE
git add configs/ppe/sources.yml src/ppe/__init__.py src/ppe/registry.py tests/test_ppe_registry.py
git commit -m "feat: add commercial PPE source registry"
```

---

### Task 2: Secure and Reproducible Source Acquisition

**Files:**
- Create: `D-FINE/src/ppe/dataset.py`
- Create: `D-FINE/tools/ppe/build_dataset.py`
- Create: `D-FINE/tests/test_ppe_dataset.py`

**Interfaces:**
- Consumes: `SourceSpec` from Task 1.
- Produces: `fetch_source(source: SourceSpec, cache_dir: Path, env: Mapping[str, str]) -> Path`
- Produces: `safe_extract_zip(archive: Path, destination: Path) -> None`
- Produces CLI: `python tools/ppe/build_dataset.py fetch --registry ... --cache ...`

- [ ] **Step 1: Write failing acquisition tests**

Cover cache hits, SHA mismatch, interrupted downloads leaving no final file, ZIP traversal such as `../../escape`, symlink entries, and missing `ROBOFLOW_API_KEY`. Mock `urllib.request.urlopen`; tests must not access the network.

- [ ] **Step 2: Run tests and verify red**

Run: `cd D-FINE && ../venv/bin/python -m unittest tests.test_ppe_dataset.AcquisitionTest -v`

Expected: missing acquisition functions.

- [ ] **Step 3: Implement resolvers with the standard library**

Use `urllib.request`, `hashlib`, `tempfile`, and `zipfile`; do not add a download SDK. Resolve Mendeley through its versioned API metadata and Roboflow through the versioned COCO export endpoint using `ROBOFLOW_API_KEY` from the environment. Stream to a temporary file, hash while downloading, rename only after validation, and reject archive entries whose resolved path escapes the destination or whose type is a symlink.

The fetch command prints source ID, resolved URL, byte count, and SHA-256. If the registry has no expected hash, it exits after download with a nonzero “review required” status; the operator adds the observed hash to the registry before `build` can consume it. This is a provenance approval step, not manual annotation.

- [ ] **Step 4: Run acquisition tests**

Run: `cd D-FINE && ../venv/bin/python -m unittest tests.test_ppe_dataset.AcquisitionTest -v`

Expected: all tests pass without network access.

- [ ] **Step 5: Commit Task 2 files**

```bash
cd D-FINE
git add src/ppe/dataset.py tools/ppe/build_dataset.py tests/test_ppe_dataset.py
git commit -m "feat: fetch PPE datasets reproducibly"
```

---

### Task 3: Canonical Conversion and Provenance Ledger

**Files:**
- Modify: `D-FINE/src/ppe/dataset.py`
- Modify: `D-FINE/tools/ppe/build_dataset.py`
- Modify: `D-FINE/tests/test_ppe_dataset.py`

**Interfaces:**
- Produces: `convert_source(source: SourceSpec, extracted_dir: Path) -> list[ImageRecord]`
- Produces: `ImageRecord` and `AnnotationRecord` dataclasses.
- Produces CLI stage: `build --registry ... --cache ... --output ... --seed 20260901`.

- [ ] **Step 1: Add failing fixture conversions**

Create tiny test fixtures in temporary directories for YOLO and COCO inputs. Assert canonical `xywh` boxes, zero-based internal class IDs, normalized source paths, rejected invalid boxes, explicit `no_*` preservation, and that a missing `gloves` annotation does not create `no_gloves`.

- [ ] **Step 2: Run conversion tests and verify red**

Run: `cd D-FINE && ../venv/bin/python -m unittest tests.test_ppe_dataset.ConversionTest -v`

Expected: missing conversion API.

- [ ] **Step 3: Implement only the two required converters**

Support YOLO text and COCO JSON because the approved sources publish those formats. Use Pillow for image dimensions and reject unreadable images. Preserve original annotations for audit but materialize detector categories in this exact order:

```python
DETECTOR_CLASSES = ("person", "helmet", "vest", "gloves", "boots")
PUBLIC_STATES = ("helmet", "vest", "gloves", "boots", "no_helmet", "no_vest", "no_gloves", "no_boots")
```

Write CSV ledgers using `csv.DictWriter`: `manifest.csv`, `rejected.csv`, and `licenses.csv`. Generate `ATTRIBUTION.md` from registry evidence without embedding untrusted source HTML.

- [ ] **Step 4: Run conversion tests**

Run: `cd D-FINE && ../venv/bin/python -m unittest tests.test_ppe_dataset.ConversionTest -v`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 3 files**

```bash
cd D-FINE
git add src/ppe/dataset.py tools/ppe/build_dataset.py tests/test_ppe_dataset.py
git commit -m "feat: canonicalize PPE annotations"
```

---

### Task 4: Deduplication, Group Split, Crops, and Audit

**Files:**
- Modify: `D-FINE/src/ppe/dataset.py`
- Modify: `D-FINE/tools/ppe/build_dataset.py`
- Modify: `D-FINE/tests/test_ppe_dataset.py`

**Interfaces:**
- Produces: `build_dataset(records, output_dir, seed, split_ratios) -> BuildReport`
- Produces immutable `annotations/instances_{train,val,test}.json` and `audit.json`.

- [ ] **Step 1: Write failing split and reproducibility tests**

Assert that exact duplicates, perceptual duplicates, a parent full frame, and all derived person crops remain in one split; the same seed produces byte-identical manifests; changing input content changes the build ID; and failed validation leaves no partially published build.

- [ ] **Step 2: Run tests and verify red**

Run: `cd D-FINE && ../venv/bin/python -m unittest tests.test_ppe_dataset.BuildTest -v`

Expected: missing build implementation.

- [ ] **Step 3: Implement the minimal deterministic builder**

Use SHA-256 for exact duplicates and Pillow 8×8 grayscale difference hashes for near-duplicate candidate grouping. Group by declared origin first, then merge duplicate-connected groups. Assign whole groups with a seeded greedy stratifier targeting 80/10/10 while minimizing source/class count error. Do not copy records to balance classes.

Generate person crops only after split assignment, using the source `person` box plus 10% bounded context. Keep full-frame and crop records; record parent ID and crop box. Write into a temporary sibling directory and atomically rename it to a build ID derived from registry hash, input hashes, seed, and builder version.

Audit must report image/origin/source/class counts, rejected reasons, exact and perceptual duplicate groups, cross-split leakage count, full-frame/crop counts, relative box-size buckets, and license counts. Abort unless cross-split leakage is zero and every materialized record has license evidence.

- [ ] **Step 4: Run builder and existing tests**

Run: `cd D-FINE && ../venv/bin/python -m unittest tests.test_ppe_dataset tests.test_det_engine -v`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 4 files**

```bash
cd D-FINE
git add src/ppe/dataset.py tools/ppe/build_dataset.py tests/test_ppe_dataset.py
git commit -m "feat: build leakage-safe PPE datasets"
```

---

### Task 5: Five-Class D-FINE-M Training Configuration

**Files:**
- Create: `D-FINE/configs/dfine/custom/ppe_detection.yml`
- Create: `D-FINE/configs/dfine/custom/dfine_hgnetv2_m_ppe.yml`
- Create: `D-FINE/tests/test_ppe_config.py`

**Interfaces:**
- Consumes: generated COCO dataset path passed through CLI updates.
- Produces: a resolved five-class, 800-pixel, batch-four fine-tuning configuration.

- [ ] **Step 1: Write a failing resolved-config test**

Load `YAMLConfig` and assert five classes, `remap_mscoco_category: false`, 800×800 train/validation resize, total batch size four, base learning rate `0.00003125`, backbone learning rate `0.000003125`, pretrained backbone disabled when `--tuning` is supplied, and output directory distinct from the legacy run.

- [ ] **Step 2: Run test and verify red**

Run: `cd D-FINE && ../venv/bin/python -m unittest tests.test_ppe_config -v`

Expected: missing config files.

- [ ] **Step 3: Add the minimal configs**

Include the upstream D-FINE-M custom config, override only dataset paths/classes, resize and multiscale base size to 800, batch size, scaled learning rates, deterministic output directory, and existing stage schedule. Do not change transformer depth, add a sampler, or move to a larger model.

The training command must use `--tuning <approved-checkpoint>` so the whole pretrained detector initializes before replacing the class head. The model registry must validate the checkpoint hash before launching.

- [ ] **Step 4: Run config and full unit tests**

Run: `cd D-FINE && ../venv/bin/python -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 5 files**

```bash
cd D-FINE
git add configs/dfine/custom/ppe_detection.yml configs/dfine/custom/dfine_hgnetv2_m_ppe.yml tests/test_ppe_config.py
git commit -m "feat: configure commercial PPE fine-tuning"
```

---

### Task 6: Visibility-Gated Eight-State PPE Policy

**Files:**
- Create: `D-FINE/src/ppe/policy.py`
- Modify: `D-FINE/src/ppe/__init__.py`
- Create: `D-FINE/tests/test_ppe_policy.py`
- Create: `D-FINE/requirements-ppe.txt`

**Interfaces:**
- Produces: `evaluate_person(person_box, detections, keypoints, config) -> PersonPPEState`
- Produces: `PersonPPEState.states: dict[str, Literal["present", "absent", "unknown"]]`
- Consumes: detector boxes/scores and BlazePose keypoints normalized to the original frame.

- [ ] **Step 1: Write failing policy tests**

Use fixed arrays, not a real pose model. Cover positive precedence, visible absence, low-confidence wrist/ankle, crop truncation, one visible glove with the other hand unknown, duplicated PPE candidates, and contradictory positive/negative source labels. Assert that low evidence always returns `unknown`, never `absent`.

- [ ] **Step 2: Run tests and verify red**

Run: `cd D-FINE && ../venv/bin/python -m unittest tests.test_ppe_policy -v`

Expected: missing policy module.

- [ ] **Step 3: Implement deterministic geometry**

Represent boxes as `(x1, y1, x2, y2)`. Use named keypoint groups for head, shoulders/hips, wrists, and ankles/heels/feet. Require all configured keypoints for a region to exceed confidence and lie within frame bounds. Match PPE centers to padded body-region rectangles, then choose the highest-score candidate. Return `unknown` for contradictory matches or unavailable regions.

Keep thresholds in a frozen `PolicyConfig`; defaults are provisional calibration values and are serialized into every evaluation report.

Add pinned compatible versions of `mediapipe` and `onnxruntime` only after installing them in the existing Python 3.12 environment succeeds. Record the BlazePose Heavy asset URL and SHA in the model registry before inference can load it.

- [ ] **Step 4: Run policy and all tests**

Run: `cd D-FINE && ../venv/bin/python -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 6 files**

```bash
cd D-FINE
git add src/ppe/__init__.py src/ppe/policy.py tests/test_ppe_policy.py requirements-ppe.txt
git commit -m "feat: infer PPE absence from visible body regions"
```

---

### Task 7: Hybrid Inference and Evaluation

**Files:**
- Create: `D-FINE/tools/ppe/run_hybrid.py`
- Create: `D-FINE/tools/ppe/evaluate_pipeline.py`
- Create: `D-FINE/tests/test_ppe_evaluation.py`

**Interfaces:**
- Produces CLI inference JSON with person boxes, positive detections, keypoint confidence, eight public labels, and internal unknown states.
- Produces `metrics.json` and `metrics.md` for detector and end-to-end comparison.

- [ ] **Step 1: Write failing aggregation tests**

Create a small prediction/ground-truth fixture with present, absent, and unknown states. Assert per-label precision/recall/F1, macro-F1, no-PPE macro recall, false alarms per person, unknown coverage, and buckets by image kind/source/person size.

- [ ] **Step 2: Run tests and verify red**

Run: `cd D-FINE && ../venv/bin/python -m unittest tests.test_ppe_evaluation -v`

Expected: missing evaluation module.

- [ ] **Step 3: Implement inference by reusing existing D-FINE loading/postprocessing**

Do not fork model-loading logic. Import `YAMLConfig`, solver setup, and postprocessor patterns already used by `tools/inference/torch_inf.py`. Batch D-FINE full-frame inference, crop each person with the same deterministic helper as the dataset builder, run BlazePose Heavy on each crop, map landmarks back to original coordinates, and call `evaluate_person`.

For pre-cropped input, treat the whole image as the person only when `--person-crop` is explicit. JSON output must distinguish `unknown` from `no_*`; rendered overlays omit unknown warnings.

- [ ] **Step 4: Implement evaluation with stdlib aggregation and existing COCO evaluator**

Reuse the repository COCO evaluator for detector AP. Implement state metrics with integer counters and explicit zero-division behavior; do not add a metrics framework. Write the resolved thresholds, dataset build ID, checkpoint hash, and model asset hash into each report.

- [ ] **Step 5: Run unit tests and a fixture end-to-end command**

Run: `cd D-FINE && ../venv/bin/python -m unittest discover -s tests -v`

Run: `cd D-FINE && ../venv/bin/python tools/ppe/evaluate_pipeline.py --fixture tests/fixtures/ppe_eval.json --output /tmp/ppe-eval-smoke`

Expected: tests pass and `/tmp/ppe-eval-smoke/metrics.json` matches fixture expectations.

- [ ] **Step 6: Commit Task 7 files**

```bash
cd D-FINE
git add tools/ppe/run_hybrid.py tools/ppe/evaluate_pipeline.py tests/test_ppe_evaluation.py
git commit -m "feat: evaluate hybrid PPE states"
```

---

### Task 8: Build, Baseline, Train, and Compare

**Files:**
- Modify after generated review only: `D-FINE/configs/ppe/sources.yml` with observed immutable archive/model hashes.
- Generate: `dataset/PPE_COMMERCIAL/<build-id>/`
- Generate: `output/dfine_m_ppe_commercial/<run-id>/`

**Interfaces:**
- Consumes all previous tasks.
- Produces the first clean dataset, clean legacy baseline, trained checkpoint, and comparison report.

- [ ] **Step 1: Verify environment before external writes**

Run:

```bash
nvidia-smi
df -h /home/hsv/hungdv/PPE
cd D-FINE && ../venv/bin/python -m unittest discover -s tests -v
```

Expected: GPU is visible, storage is sufficient for source archives plus immutable build, and all tests pass.

- [ ] **Step 2: Fetch sources and record hashes**

Run the Mendeley fetch first. Run Roboflow only when `ROBOFLOW_API_KEY` exists. Review the printed exact URL/license evidence/hash, add that hash to `sources.yml`, rerun registry tests, then rerun fetch so it exits successfully from the verified cache.

Commands:

```bash
cd D-FINE
../venv/bin/python tools/ppe/build_dataset.py fetch --registry configs/ppe/sources.yml --cache ../dataset/.cache/ppe
../venv/bin/python -m unittest tests.test_ppe_registry -v
```

- [ ] **Step 3: Build the commercial dataset twice to verify determinism**

```bash
cd D-FINE
../venv/bin/python tools/ppe/build_dataset.py build --registry configs/ppe/sources.yml --cache ../dataset/.cache/ppe --output ../dataset/PPE_COMMERCIAL --seed 20260901
../venv/bin/python tools/ppe/build_dataset.py verify --dataset ../dataset/PPE_COMMERCIAL/current
```

Expected: zero cross-split origin/exact/perceptual leakage, no unapproved licenses, and the second build resolves to the same build ID without rewriting it.

- [ ] **Step 4: Establish the clean baseline**

Run the legacy checkpoint on the new clean test set and write a report under `output/dfine_m_ppe_commercial/baseline-legacy`. Missing `person` output or ontology incompatibility must be recorded explicitly; do not compare incompatible aggregate COCO mAP as if it were equivalent.

- [ ] **Step 5: Fine-tune D-FINE-M from the approved checkpoint**

Use a new output directory and never resume the legacy run:

```bash
cd D-FINE
export PPE_APPROVED_CHECKPOINT=/home/hsv/hungdv/PPE/models/approved/dfine_m_coco.pth
CUDA_VISIBLE_DEVICES=0 ../venv/bin/python train.py \
  -c configs/dfine/custom/dfine_hgnetv2_m_ppe.yml \
  -t "$PPE_APPROVED_CHECKPOINT" \
  --use-amp --seed 20260901 \
  --output-dir ../output/dfine_m_ppe_commercial/run-800
```

Expected: resolved config shows five classes, batch four, 800-pixel transforms, scaled learning rates, and the approved tuning checkpoint hash. Monitor validation AP; stop only on completion or a concrete failure, not because an individual epoch fluctuates.

- [ ] **Step 6: Calibrate policy on validation and freeze thresholds**

Search a small deterministic grid over detector confidence, keypoint confidence, and region padding. Select the configuration maximizing macro-F1 subject to per-label unknown-coverage reporting. Save it once as `policy.json`; never tune on test.

- [ ] **Step 7: Evaluate once on the clean test set**

Generate detector AP and end-to-end state metrics for full-frame/crop, source, person size, and occlusion buckets. Compare against the clean legacy baseline. Mark `no_gloves` or `no_boots` as insufficient-data when ground-truth support is too small; do not manufacture a percentage.

- [ ] **Step 8: Run final verification**

```bash
cd D-FINE
../venv/bin/python -m unittest discover -s tests -v
../venv/bin/python tools/ppe/build_dataset.py verify --dataset ../dataset/PPE_COMMERCIAL/current
git status --short
```

Expected: all tests pass, dataset verification passes, generated reports contain source/build/model hashes, and the pre-existing user modifications remain present.

- [ ] **Step 9: Commit only the reviewed registry hashes and documentation**

```bash
cd D-FINE
git add configs/ppe/sources.yml
git commit -m "chore: lock approved PPE asset hashes"
```

Do not commit downloaded datasets, model weights, output checkpoints, caches, or the user's unrelated dirty files.

