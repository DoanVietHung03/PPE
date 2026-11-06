# PPE Detection

Phát hiện trang bị bảo hộ (PPE) bằng D-FINE / RF-DETR.

## Dữ liệu & weights

Dataset, weights và kết quả training không nằm trong repo, tải tại Google Drive:
https://drive.google.com/drive/folders/1al1n9cvc8xVdHq3jdqq1GtqbPcXvuqiZ

| Drive | Giải nén / đặt vào |
|---|---|
| `models/` | `models/` |
| `output/` | `output/` |
| `datasets/dataset-ppe-5c-current-6a4dd4f1292433d7.tar` | `tar -xf <file>` tại thư mục gốc (bản dataset đang dùng) |
| `datasets/dataset-ppe-5c-other-versions.tar` | `tar -xf <file>` (các phiên bản ppe-5c khác) |
| `datasets/dataset-candidates.tar` | `tar -xf <file>` (ppe-13c, pants-2c) |
| `datasets/dataset-cache.tar` | `tar -xf <file>` (dữ liệu nguồn thô) |
| `datasets/archive-legacy-8c.tar` | `tar -xf <file>` (bản 8 class cũ) |

Sau khi giải nén, tạo lại các symlink nếu cần:

```bash
ln -s ppe-5c/dfine_m_960.onnx models/ppe_dfine_m_960.onnx
ln -s ppe-5c dataset/PPE_COMMERCIAL
ln -s ppe-5c/training output/dfine_m_ppe_commercial
ln -s ppe-5c/evaluation output/hybrid-eval
```

## Cấu trúc

- `D-FINE/`: D-FINE (đã chỉnh sửa) + code PPE (`src/ppe`, `tools/ppe`, `configs/ppe`)
- `RF-DETR-L/`: train RF-DETR Large
- `pipelines/`: pipeline cho ppe-5c, ppe-9c, candidates
- `docs/`: tài liệu class, dataset, deploy ONNX
