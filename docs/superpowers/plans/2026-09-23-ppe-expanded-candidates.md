# PPE Expanded Candidates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build reproducible, separately trainable 13-class PPE and 2-class pants candidate datasets from the audited archives without changing the production five-class dataset.

**Architecture:** Parameterize the existing content-addressed builder with explicit detector/state/context taxonomies, then add one candidate CLI path that selects exactly one pinned registry source. Reuse existing acquisition, conversion, deduplication, split grouping, audit, and `CURRENT` publication code; training configs consume only verified candidate outputs.

**Tech Stack:** Python 3.12, unittest/pytest, Pillow, PyYAML, COCO JSON, D-FINE YAML configs.

**Spec:** `docs/superpowers/specs/2026-09-23-ppe-expanded-candidates-design.md`

## Global Constraints

- Preserve `dataset/ppe-5c`, `pipelines/ppe-5c`, its model and deployment defaults byte-for-byte.
- Never combine A and B into a conventional 15-class COCO dataset.
- Do not consume source C (`ultralytics_construction_ppe`) in either new candidate.
- `unknown` is a policy state, never a detector category.
- Use only the pinned A/B archives and retain their SHA-256/license evidence.
- Reuse the existing builder; no parallel builder implementation or new dependency.

## Review Focus

- A category mapped by the registry but absent from the selected taxonomy must be rejected from output, never assigned an accidental ID; Task 1 tests this.
- Duplicate/variant images originating in different source splits must remain in one generated split; Task 1 tests origin grouping with custom taxonomies.
- Candidate selection must load exactly its declared source, never every approved source; Task 2 tests source isolation.
- An absent or checksum-mismatched pinned archive must fail without publishing `CURRENT`; Task 2 tests failure atomicity.
- Every candidate category must have train/val/test annotations before a training config is considered usable; Task 4 verifies this against built artifacts.

---

### Task 1: Parameterize the Existing Dataset Builder

**Files:**
- Modify: `D-FINE/src/ppe/build.py`
- Test: `D-FINE/tests/test_ppe_dataset.py`

**Interfaces:**
- Consumes: existing `build_dataset(records, output_dir, seed, *, rejections, licenses)` callers.
- Produces: `build_dataset(..., detector_classes=DETECTOR_CLASSES, state_classes=PUBLIC_STATES, context_classes=("gloves", "boots")) -> BuildReport`.

- [ ] **Step 1: Write failing tests for custom category order and exclusion**

Add a test that builds one synthetic record with `person`, `mask`, `no_mask`, and `helmet`, calls:

```python
report = build_dataset(
    [record],
    root / "output",
    seed=7,
    detector_classes=("person", "helmet", "mask", "no_mask"),
    state_classes=("helmet", "mask", "no_mask"),
    context_classes=(),
)
```

Assert `instances_train.json` categories preserve that exact order and all four annotations use valid IDs. Add an extra `long_pants` annotation and assert it is absent from detector/state output.

- [ ] **Step 2: Run the focused test and observe RED**

Run: `cd D-FINE && ../venv/bin/python -m unittest tests.test_ppe_dataset.BuildTest.test_custom_taxonomy_controls_output_categories -v`

Expected: ERROR because `build_dataset` does not accept `detector_classes`.

- [ ] **Step 3: Implement the minimal builder parameters**

Add keyword-only tuple parameters with current constants as defaults. Inside `build_dataset`, convert them to tuples once, reject empty detector taxonomies and duplicate names, then replace all hard-coded `DETECTOR_CLASSES`, `PUBLIC_STATES`, and `{gloves, boots}` membership/index operations in the build path with the provided values. Include all three tuples in `_canonical_build_id` input so different taxonomies cannot share a build ID.

- [ ] **Step 4: Test grouping remains leak-free under a custom taxonomy**

Extend the existing split/dedup test with two records sharing `origin_group` but different `split_hint`; build with `detector_classes=("mask",)` and assert their manifest rows have one generated split and `cross_split_duplicate_groups == 0`.

- [ ] **Step 5: Run Task 1 tests**

Run: `cd D-FINE && ../venv/bin/python -m unittest tests.test_ppe_dataset -v`

Expected: all dataset acquisition/conversion/build tests pass.

- [ ] **Step 6: Commit checkpoint when Git scope permits**

```bash
git -C D-FINE add src/ppe/build.py tests/test_ppe_dataset.py
git -C D-FINE commit -m "feat: parameterize PPE dataset taxonomy"
```

If the shared dirty D-FINE worktree prevents a safe isolated commit, record that ruling in the execution ledger and do not include unrelated changes.

### Task 2: Add Candidate-Specific Build Command

**Files:**
- Modify: `D-FINE/tools/ppe/build_dataset.py`
- Modify: `D-FINE/tests/test_ppe_build_cli.py`
- Modify: `D-FINE/tests/test_ppe_registry.py`

**Interfaces:**
- Consumes: Task 1 `build_dataset` taxonomy keyword arguments and pinned registry sources `roboflow_ppe_compliance_v2`, `roboflow_pants_v2_v1`.
- Produces: CLI `build-candidate --candidate {ppe-13c,pants-2c} --registry PATH --cache PATH --output PATH --seed INT`.

- [ ] **Step 1: Write a failing candidate-selection test**

Patch `fetch_source`, `_extracted_root`, `convert_source`, and `build_dataset` at the module boundary. Invoke `main([...])` with `build-candidate --candidate ppe-13c`; assert conversion receives only `roboflow_ppe_compliance_v2` and `build_dataset` receives:

```python
detector_classes=(
    "person", "helmet", "no_helmet", "vest", "no_vest", "gloves",
    "no_gloves", "boots", "no_boots", "mask", "no_mask",
    "eye_protection", "no_eye_protection",
)
```

Add the corresponding `pants-2c` assertion for `("long_pants", "short_pants")`.

- [ ] **Step 2: Run focused CLI tests and observe RED**

Run: `cd D-FINE && ../venv/bin/python -m unittest tests.test_ppe_build_cli -v`

Expected: FAIL because `build-candidate` is not registered.

- [ ] **Step 3: Implement candidate constants and command**

Use one module-level dictionary, not a new class hierarchy:

```python
CANDIDATES = {
    "ppe-13c": {
        "source": "roboflow_ppe_compliance_v2",
        "detector_classes": (...13 names...),
        "state_classes": (...12 names excluding person...),
        "context_classes": ("gloves", "boots"),
    },
    "pants-2c": {
        "source": "roboflow_pants_v2_v1",
        "detector_classes": ("long_pants", "short_pants"),
        "state_classes": ("long_pants", "short_pants"),
        "context_classes": (),
    },
}
```

Resolve the source by exact ID regardless of `approved_sources()`, require its status to be `pending_clearance` and its SHA-256 to be present, verify/extract/convert only that archive, call Task 1's builder, then call `_publish_current` only after success.

- [ ] **Step 4: Test missing/checksum-mismatched archives do not publish**

Make `fetch_source` raise `ValueError("cached checksum mismatch")`; assert neither `CURRENT` nor `current` exists in the temporary output directory.

- [ ] **Step 5: Pin registry class order and archive hashes**

Extend `test_extended_ppe_candidates_are_pinned_and_mapped` to assert both full mapping value sets and the exact SHA-256 values from the audit report.

- [ ] **Step 6: Run Task 2 tests**

Run: `cd D-FINE && ../venv/bin/python -m unittest tests.test_ppe_build_cli tests.test_ppe_registry -v`

Expected: all tests pass.

- [ ] **Step 7: Commit checkpoint when Git scope permits**

```bash
git -C D-FINE add tools/ppe/build_dataset.py tests/test_ppe_build_cli.py tests/test_ppe_registry.py
git -C D-FINE commit -m "feat: build isolated PPE dataset candidates"
```

### Task 3: Build and Verify Both Candidate Datasets

**Files:**
- Create: `dataset/candidates/ppe-13c/<content-id>/` plus `CURRENT` and `current`
- Create: `dataset/candidates/pants-2c/<content-id>/` plus `CURRENT` and `current`
- Modify: `docs/research/2026-09-23-ppe-expanded-dataset-audit.md`

**Interfaces:**
- Consumes: Task 2 CLI and pinned archives in `dataset/.cache/ppe`.
- Produces: verified COCO `instances_{train,val,test}.json`, state JSON, manifests, license files, attribution and audit reports.

- [ ] **Step 1: Build the 13-class candidate**

Run:

```bash
venv/bin/python D-FINE/tools/ppe/build_dataset.py build-candidate \
  --candidate ppe-13c --registry D-FINE/configs/ppe/sources.yml \
  --cache dataset/.cache/ppe --output dataset/candidates/ppe-13c --seed 20260923
```

Expected: one immutable build and an atomic `current` pointer; production `dataset/ppe-5c/CURRENT` remains unchanged.

- [ ] **Step 2: Build the pants candidate**

Run the same command with `--candidate pants-2c` and output `dataset/candidates/pants-2c`.

- [ ] **Step 3: Verify both builds**

Run `build_dataset.py verify` on both `current` paths. Expected: zero cross-split duplicate groups.

- [ ] **Step 4: Audit class coverage and provenance**

Run a stdlib JSON check asserting category order, nonzero train/val/test count per category, source IDs, SHA-256 evidence, and `CC-BY-4.0` attribution. Assert neither manifest contains `ultralytics_construction_ppe`.

- [ ] **Step 5: Update the research audit**

Append generated build IDs, post-dedup image/annotation counts, per-split class counts, rejection counts, and explicit confirmation that `dataset/ppe-5c/CURRENT` did not change.

### Task 4: Add Candidate Training Configs and Final Verification

**Files:**
- Create: `pipelines/candidates/ppe-13c/configs/dataset.yml`
- Create: `pipelines/candidates/ppe-13c/configs/dfine_m_960.yml`
- Create: `pipelines/candidates/pants-2c/configs/dataset.yml`
- Create: `pipelines/candidates/pants-2c/configs/dfine_m_960.yml`
- Create: `pipelines/candidates/README.md`
- Modify: `D-FINE/tests/test_ppe_config.py`

**Interfaces:**
- Consumes: Task 3 candidate `current` COCO paths.
- Produces: resolvable D-FINE configs with `num_classes` 13 and 2; no production deployment changes.

- [ ] **Step 1: Write failing config-resolution tests**

Load both model configs through `YAMLConfig`; assert `num_classes`, 960 spatial size, batch size 2, train/val annotation paths, and candidate-specific output directories. Assert production config remains five classes.

- [ ] **Step 2: Run config tests and observe RED**

