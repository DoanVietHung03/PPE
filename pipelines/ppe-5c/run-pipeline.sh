#!/usr/bin/env bash
set -euo pipefail

PPE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DFINE_ROOT="$PPE_ROOT/D-FINE"
PPE_PYTHON="$PPE_ROOT/venv/bin/python"
PPE_REGISTRY="$DFINE_ROOT/configs/ppe/sources.yml"
PPE_CACHE="$PPE_ROOT/dataset/.cache/ppe"
PPE_DATASET="$PPE_ROOT/dataset/ppe-5c"
PPE_CHECKPOINT="$PPE_ROOT/models/approved/dfine_m_coco.pth"

if [[ -z "${ROBOFLOW_API_KEY:-}" && -f "$PPE_ROOT/.env" ]]; then
  while IFS='=' read -r key value; do
    [[ "$key" == "ROBOFLOW_API_KEY" ]] && ROBOFLOW_API_KEY=$value
  done < "$PPE_ROOT/.env"
fi
if [[ -z "${ROBOFLOW_API_KEY:-}" ]]; then
  echo "Missing ROBOFLOW_API_KEY. Run: $PPE_ROOT/pipelines/ppe-5c/setup-credentials.sh" >&2
  exit 1
fi
export ROBOFLOW_API_KEY

cd "$DFINE_ROOT"
"$PPE_PYTHON" tools/ppe/verify_model.py \
  --registry "$PPE_REGISTRY" --model dfine_m_coco --path "$PPE_CHECKPOINT"
set +e
"$PPE_PYTHON" tools/ppe/build_dataset.py fetch \
  --registry "$PPE_REGISTRY" --cache "$PPE_CACHE" \
  --source roboflow_construction_site_safety_v30
fetch_status=$?
set -e
if [[ $fetch_status -eq 2 ]]; then
  echo "Archive downloaded and held for one-time SHA/provenance review; training was not started." >&2
  exit 2
fi
[[ $fetch_status -eq 0 ]] || exit "$fetch_status"

"$PPE_PYTHON" tools/ppe/build_dataset.py build \
  --registry "$PPE_REGISTRY" --cache "$PPE_CACHE" \
  --output "$PPE_DATASET" --seed 20260901
"$PPE_PYTHON" tools/ppe/build_dataset.py verify --dataset "$PPE_DATASET/current"
"$PPE_PYTHON" -m unittest discover -s tests -p 'test_*.py' -q

build_id=$(<"$PPE_DATASET/CURRENT")
output="$PPE_ROOT/output/ppe-5c/training/run-960-$build_id"
if [[ -e "$output" ]]; then
  echo "Refusing to overwrite existing run: $output" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 "$PPE_PYTHON" train.py \
  -c ../pipelines/ppe-5c/configs/dfine_m_960.yml \
  -t "$PPE_CHECKPOINT" --use-amp --seed 20260901 --output-dir "$output"
