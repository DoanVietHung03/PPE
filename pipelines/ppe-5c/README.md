# PPE 5-class pipeline

Detector classes: person, helmet, vest, gloves, boots.

The hybrid policy derives no_helmet, no_vest, no_gloves, and no_boots only when
the corresponding body region is visible. D-FINE remains in the repository
root as the shared training framework.

Canonical paths:

- Config: pipelines/ppe-5c/configs/dfine_m_960.yml
- Dataset: dataset/ppe-5c/current
- Models: models/ppe-5c
- Training output: output/ppe-5c/training
- Evaluation output: output/ppe-5c/evaluation

Run from the repository root:

```bash
./pipelines/ppe-5c/setup-credentials.sh
./pipelines/ppe-5c/run-pipeline.sh
venv/bin/python pipelines/ppe-5c/tools/run_hybrid.py --help
venv/bin/python pipelines/ppe-5c/tools/evaluate_pipeline.py --help
```
