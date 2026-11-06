# PPE 5c/9c Layout Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Give the working five-class hybrid pipeline one canonical folder and reserve a documentation-only boundary for a future nine-class detector.

**Architecture:** Move five-class entry points and configuration under pipelines/ppe-5c while leaving D-FINE as the shared framework. Rename large artifacts in place under dataset/ppe-5c, models/ppe-5c, and output/ppe-5c, with compatibility symlinks for old external artifact paths only.

**Tech Stack:** Bash, Python 3.12, D-FINE/PyTorch, ONNX Runtime, MediaPipe, YAML, unittest.

**Spec:** docs/superpowers/specs/2026-09-22-ppe-pipeline-layout-design.md

## Global Constraints

- ppe-5c means person, helmet, vest, gloves, and boots; no_* remains pose-gated policy output.
- ppe-9c means ppe-5c plus no_helmet, no_vest, no_gloves, and no_boots as detector outputs.
- D-FINE remains shared and is not copied.
- Do not duplicate the 6.5 GB dataset or model/checkpoint files.
- Preserve the existing validation and test metrics byte-for-byte.
- Root is not a Git repository; do not invent commit steps. D-FINE has unrelated dirty/untracked work that must be preserved.
- Use recoverable same-filesystem moves and create compatibility symlinks only after validating their targets.

## Review Focus

- Running the moved deploy script directly must still resolve its local ppe package and canonical model assets.
- Running the moved tools from any working directory must still import D-FINE/src.
- YAML include paths must resolve from pipelines/ppe-5c/configs and retain num_classes: 5.
- Old external dataset/model/output artifact paths must resolve through valid symlinks.
- No live command, test, or current documentation may point at the legacy eight-class config as production.

---

### Task 1: Lock the canonical path contract with tests

**Files:**
- Create: D-FINE/tests/test_ppe_layout.py
- Modify: D-FINE/tests/test_ppe_config.py
- Modify: D-FINE/tests/test_deploy_pipeline.py

**Interfaces:**
- Consumes: the approved canonical layout from the spec.
- Produces: regression checks for canonical files, artifact symlinks, imports, and resolved five-class YAML.

- [ ] **Step 1: Write failing layout tests**

Add tests that assert these paths exist:

    pipelines/ppe-5c/configs/dfine_m_960.yml
    pipelines/ppe-5c/deploy/pipeline.py
    pipelines/ppe-5c/tools/run_hybrid.py
    pipelines/ppe-5c/tools/evaluate_pipeline.py
    pipelines/ppe-9c/README.md

Assert dataset/PPE_COMMERCIAL, output/dfine_m_ppe_commercial, and output/hybrid-eval are symlinks whose resolved targets live below dataset/ppe-5c or output/ppe-5c. Assert the moved config resolves to num_classes 5 and 960x960.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

    cd D-FINE
    ../venv/bin/python -m unittest tests.test_ppe_layout tests.test_ppe_config tests.test_deploy_pipeline -v

Expected: failure because pipelines/ppe-5c does not exist.

- [ ] **Step 3: Update test imports to the new deploy/tool locations**

Insert pipelines/ppe-5c on sys.path in test_deploy_pipeline.py before importing deploy.pipeline. Point test_ppe_config.py at ../pipelines/ppe-5c/configs/dfine_m_960.yml.

- [ ] **Step 4: Leave tests red until Tasks 2 and 3 create the paths**

No production compatibility wrapper is added at the old source-code locations.

### Task 2: Move the five-class source entry points

**Files:**
- Move: deploy/ to pipelines/ppe-5c/deploy/
- Move: D-FINE/tools/ppe/run_hybrid.py to pipelines/ppe-5c/tools/run_hybrid.py
- Move: D-FINE/tools/ppe/evaluate_pipeline.py to pipelines/ppe-5c/tools/evaluate_pipeline.py
- Move: run-ppe-pipeline.sh to pipelines/ppe-5c/run-pipeline.sh
- Create: pipelines/ppe-5c/README.md
- Create: pipelines/ppe-9c/README.md

**Interfaces:**
- Consumes: D-FINE/src/ppe modules and root artifact paths.
- Produces: direct CLI entry points under pipelines/ppe-5c and taxonomy documentation for both model families.

- [ ] **Step 1: Move source paths without copying**

Use mkdir and mv with explicit targets. Before every move, assert the source exists and destination does not. Do not move D-FINE/src/ppe or D-FINE/tools/ppe/build_dataset.py.

- [ ] **Step 2: Fix D-FINE imports in moved tools**

Set the D-FINE root from the moved script path:

    ROOT = Path(__file__).resolve().parents[3]
    DFINE_ROOT = ROOT / "D-FINE"
    sys.path.insert(0, str(DFINE_ROOT))

Keep deploy/pipeline.py using its local relative ppe package.

- [ ] **Step 3: Update run-pipeline.sh roots**

Resolve repository root with three parent traversals from pipelines/ppe-5c/run-pipeline.sh. Invoke moved configs/tools by canonical paths and keep build_dataset.py in D-FINE/tools/ppe.

- [ ] **Step 4: Write concise READMEs**

The ppe-5c README lists its five detector classes, explains pose-derived no_* states, and provides canonical train, inference, and evaluation commands. The ppe-9c README lists all nine classes and states that no implementation/checkpoint exists yet.

- [ ] **Step 5: Run syntax/import checks**

Run:

    venv/bin/python -m py_compile pipelines/ppe-5c/deploy/pipeline.py pipelines/ppe-5c/tools/run_hybrid.py pipelines/ppe-5c/tools/evaluate_pipeline.py
    venv/bin/python pipelines/ppe-5c/deploy/pipeline.py --help
    venv/bin/python pipelines/ppe-5c/tools/run_hybrid.py --help
    venv/bin/python pipelines/ppe-5c/tools/evaluate_pipeline.py --help

Expected: exit 0 for all commands.

### Task 3: Consolidate five-class configs

**Files:**
- Move/merge: D-FINE/configs/dfine/custom/ppe_detection.yml to pipelines/ppe-5c/configs/dataset.yml
- Move/merge: D-FINE/configs/dfine/custom/dfine_hgnetv2_m_ppe_960.yml to pipelines/ppe-5c/configs/dfine_m_960.yml
- Archive: D-FINE/configs/dfine/custom/dfine_hgnetv2_m_ppe.yml
- Modify: D-FINE/tools/ppe/analyze_detection.py
- Modify: D-FINE/tests/test_ppe_config.py

**Interfaces:**
- Consumes: D-FINE/configs/dfine/custom/dfine_hgnetv2_m_custom.yml.
- Produces: one canonical five-class 960px training config.

- [ ] **Step 1: Move the 960px config and dataset config**

Change includes in dfine_m_960.yml to explicit paths relative to its new directory. Change dataset paths to dataset/ppe-5c/current and output_dir to output/ppe-5c/training.

- [ ] **Step 2: Remove the superseded 800px live config**

Move it with the legacy material in Task 5 rather than maintaining two active five-class configs.

- [ ] **Step 3: Update config consumers**

Point analyze_detection.py, run-pipeline.sh, tests, and README commands to pipelines/ppe-5c/configs/dfine_m_960.yml.

- [ ] **Step 4: Run config tests**

Run:

    cd D-FINE
    ../venv/bin/python -m unittest tests.test_ppe_config tests.test_ppe_detection_analysis -v

Expected: PASS and resolved num_classes equals 5.

### Task 4: Rename large artifacts in place

**Files:**
- Move: dataset/PPE_COMMERCIAL to dataset/ppe-5c
- Move: models/ppe_dfine_m_960.onnx to models/ppe-5c/dfine_m_960.onnx
- Move: models/approved/pose_landmarker_heavy.task to models/ppe-5c/pose_landmarker_heavy.task
- Move: output/dfine_m_ppe_commercial to output/ppe-5c/training
- Move: output/hybrid-eval to output/ppe-5c/evaluation
- Replace duplicates: pipelines/ppe-5c/deploy/models/* with symlinks to models/ppe-5c/*

**Interfaces:**
- Produces: canonical artifact roots and compatibility aliases for existing external commands.

- [ ] **Step 1: Record integrity before moves**

Record real paths, sizes, and SHA-256 for both deployable model assets. Record checksums for existing metrics.json files.

- [ ] **Step 2: Move each artifact atomically**

Create only the exact parent directories, then use mv for each explicit source. Never use recursive deletion or unresolved globs.

- [ ] **Step 3: Validate targets before aliases**

Verify dataset/ppe-5c/current resolves, the ONNX and pose hashes match Step 1, checkpoints exist under output/ppe-5c/training, and metrics exist under output/ppe-5c/evaluation.

- [ ] **Step 4: Create compatibility symlinks**

Create:

    dataset/PPE_COMMERCIAL -> ppe-5c
    output/dfine_m_ppe_commercial -> ppe-5c/training
    output/hybrid-eval -> ppe-5c/evaluation
    models/ppe_dfine_m_960.onnx -> ppe-5c/dfine_m_960.onnx
    models/approved/pose_landmarker_heavy.task -> ../ppe-5c/pose_landmarker_heavy.task

The moved deploy/models entries point to the same canonical model files.

- [ ] **Step 5: Verify integrity after moves**

Re-run the hashes and metrics checksums. They must match Step 1 exactly. Run test_ppe_layout.py and require all symlink assertions to pass.

### Task 5: Archive the legacy eight-class experiment

**Files:**
- Move: configs/dfine/ to archive/legacy-8c/configs/
- Move: D-FINE/NaN.pth to archive/legacy-8c/NaN.pth
- Move: dataset/PPE_DFINE to archive/legacy-8c/dataset
- Move: superseded D-FINE five-class 800px config to archive/superseded-5c/
- Create: archive/legacy-8c/README.md

**Interfaces:**
- Produces: no live eight-class production path, while preserving the failed experiment and provenance.

- [ ] **Step 1: Write archive metadata**

Document the exact eight labels, absence of person, checkpoint date, 49,632 non-finite values, and the fact that the checkpoint is unusable.

- [ ] **Step 2: Move explicit legacy paths**

Use same-filesystem mv operations. Do not create compatibility symlinks for legacy configs, dataset, or NaN.pth.

- [ ] **Step 3: Prove no production consumer references legacy paths**

Run:

    rg -n "PPE_DFINE|configs/dfine/ppe_detection|NaN\\.pth" pipelines D-FINE docs --glob '!docs/superpowers/plans/2026-09-01-commercial-ppe-pipeline.md'

Expected: only archive/history documentation references remain.

### Task 6: Update current documentation and path consumers

**Files:**
- Modify: docs/DEPLOY_ONNX_NVIDIA.md
- Modify: docs/research/2026-09-14-ppe-boots-gloves-datasets.md
- Modify: D-FINE/configs/ppe/sources.yml
- Modify: all current tests/scripts returned by the path scan
- Preserve unchanged: historical specs/plans and docs/DEPLOY_ONNX_NVIDIA.md.orig

**Interfaces:**
- Consumes: canonical paths created in Tasks 2-5.
- Produces: current operational documentation with no stale live commands.

- [ ] **Step 1: Replace operational paths**

Use exact canonical paths in current docs and scripts. Historical plans/specs keep old paths as historical evidence and are not presented as runnable current commands.

- [ ] **Step 2: Update provenance evidence paths**

Point registry evidence formerly under dataset/PPE_DFINE to archive/legacy-8c/dataset/audit.json. Keep source status quarantined.

- [ ] **Step 3: Scan for stale live references**

Run the same rg path query used during planning. Classify every remaining match as historical, compatibility alias, or bug; fix every bug.

### Task 7: Full verification and handoff

**Files:**
- Verify all files changed by Tasks 1-6.
- Do not modify validation/test prediction or metric contents.

**Interfaces:**
- Produces: verified canonical five-class pipeline and documentation-only nine-class boundary.

- [ ] **Step 1: Run the full test suite**

Run:

    cd D-FINE
    ../venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v

Expected: all tests PASS.

- [ ] **Step 2: Verify Python entry points**

Run py_compile and --help commands from Task 2. Expected: exit 0.

- [ ] **Step 3: Run one-image deploy smoke test**

Use pipelines/ppe-5c/deploy/img_test/test_1.jpg with the canonical ONNX and pose assets. Write JSON/media under /tmp and require exit 0 plus non-empty JSON.

- [ ] **Step 4: Verify artifact integrity and metrics preservation**

Compare hashes captured before migration. Check that validation/test metrics JSON files are byte-identical and accessible through canonical paths.

- [ ] **Step 5: Final stale-path scan**

No current executable/config/test may rely on an old source path. Compatibility aliases may appear only in migration documentation and symlink assertions.

- [ ] **Step 6: Report migration**

List canonical entry points, compatibility symlinks, archived legacy material, test count, smoke-test result, and any unrelated pre-existing D-FINE changes left untouched.

