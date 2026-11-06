# PPE candidate training

These experiments are independent and do not replace `pipelines/ppe-5c`.

From `D-FINE`, train the 13-class PPE candidate:

```bash
../venv/bin/python train.py \
  -c ../pipelines/candidates/ppe-13c/configs/dfine_m_960.yml \
  -t ../models/approved/dfine_m_coco.pth \
  --use-amp --seed 20260923
```

Train the pants candidate:

```bash
../venv/bin/python train.py \
  -c ../pipelines/candidates/pants-2c/configs/dfine_m_960.yml \
  -t ../models/approved/dfine_m_coco.pth \
  --use-amp --seed 20260923
```

Evaluate the models independently. Do not combine their datasets into one
15-class detector until the missing cross-source annotations are completed.
