# Bộ class và nguồn dữ liệu cho bài toán kiểm tra PPE

*Cập nhật: 23/09/2026. Mục tiêu: object detection để kiểm tra trang phục tại nhà máy, xưởng hoặc công trường.*

## 1. Danh sách 15 class đề xuất

| ID đề xuất | Class thống nhất | Ý nghĩa / điều kiện gán nhãn | Nhãn trong nguồn dữ liệu |
|---:|---|---|---|
| 0 | `person` | Người cần gắn các kết quả kiểm tra PPE | `Person` (A) |
| 1 | `helmet` | Đầu có đội mũ bảo hộ | `Helmet` (A) |
| 2 | `no_helmet` | Nhìn rõ đầu và xác nhận không có mũ bảo hộ | `NO-Helmet` (A) |
| 3 | `vest` | Thân trên mặc áo bảo hộ/áo phản quang theo quy định | `Vest` (A) |
| 4 | `no_vest` | Nhìn rõ thân trên và xác nhận không mặc áo theo quy định | `NO-Vest` (A) |
| 5 | `gloves` | Tay có găng bảo hộ | `Gloves` (A) |
| 6 | `no_gloves` | Nhìn rõ tay và xác nhận không có găng bảo hộ | `NO-Gloves` (A) |
| 7 | `boots` | Chân có giày/ủng bảo hộ theo quy định | `Boots` (A) |
| 8 | `no_boots` | Nhìn rõ giày/chân và xác nhận không có loại bảo hộ theo quy định | `NO-Boots` (A) |
| 9 | `mask` | Mặt đeo đúng loại khẩu trang được chấp nhận | `Mask` (A) |
| 10 | `no_mask` | Nhìn rõ mặt và xác nhận không đeo khẩu trang theo quy định | `NO-Mask` (A) |
| 11 | `eye_protection` | Đeo kính bảo hộ được chấp nhận | `Goggles` (A); `goggles` (C) |
| 12 | `no_eye_protection` | Nhìn rõ vùng mắt và xác nhận không có kính bảo hộ được chấp nhận | `NO-Goggles` (A); `no_goggle` (C) |
| 13 | `long_pants` | Mặc quần dài đạt quy định | `long-pants` (B) |
| 14 | `short_pants` | Mặc quần ngắn; vi phạm **nếu** nội quy bắt buộc quần dài | `short-pants` (B) |

**Vì sao không đặt `pants/no_pants`?** Quần short vẫn là quần. Nếu cần cảnh báo «không mặc quần dài», hãy dùng `short_pants` như bằng chứng *một dạng* vi phạm, hoặc đặt trạng thái nghiệp vụ `no_long_pants` sau bước kiểm tra. `short_pants` không bao phủ các trường hợp như váy, quần bị che khuất hay trang phục đặc thù. Cần bổ sung ảnh thực tế và định nghĩa riêng cho những trường hợp này.

`eye_protection` gộp kính bảo hộ dạng ôm mắt và kính bảo hộ thông thường **chỉ khi** quy định của bạn chấp nhận cả hai; nguồn A gọi chung là `Goggles`, cần kiểm tra ảnh để chắc chắn phạm vi gán nhãn phù hợp. Nếu chỉ chấp nhận goggles, có thể giữ nguyên `goggles/no_goggles`.

`unknown` là **trạng thái khi không thể quan sát/không đủ chắc chắn**, không phải class thứ 16 tự động gán vào YOLO. Đừng biến một vùng ngoài khung hình, bị che, hoặc quá mờ thành nhãn `no_*`.

## 2. Đường dẫn tải dữ liệu

### A — PPE Detection and Compliance, nguồn chính cho 13 class

- [Trang bộ dữ liệu](https://universe.roboflow.com/izanagi/ppe-detection-and-compliance) · [phiên bản v2 để tải](https://universe.roboflow.com/izanagi/ppe-detection-and-compliance/dataset/2).
- 13 class công bố: `Helmet`, `NO-Helmet`, `Mask`, `NO-Mask`, `Vest`, `NO-Vest`, `Goggles`, `NO-Goggles`, `Boots`, `NO-Boots`, `Gloves`, `NO-Gloves`, `Person`. Có **10.080 ảnh trong bản v2**; trang tổng quan hiển thị 10.081 ảnh (hai chỉ số có phạm vi khác nhau).
- Giấy phép được trang dự án ghi: **CC BY 4.0**. Tải trên trang phiên bản bằng nút **Download → YOLOv8** (hoặc định dạng YOLO bạn dùng); có thể cần đăng nhập Roboflow. Đây là trang tải theo phiên bản, không phải URL ZIP công khai cố định.
- Nguồn này cung cấp toàn bộ ID **0–12** sau khi đổi tên class. Nó **không có** `long_pants` hay `short_pants`.

### B — Pants_v2, nguồn cho 2 class quần

- [Trang bộ dữ liệu](https://universe.roboflow.com/nckh-oufrk/pants_v2) · [phiên bản v1 để tải](https://universe.roboflow.com/nckh-oufrk/pants_v2/dataset/1).
- **796 ảnh**, 2 class: `long-pants`, `short-pants`; tương ứng ID **13–14**. Trang dự án ghi giấy phép **CC BY 4.0**.
- Tải bằng **Download → YOLOv8** trên trang phiên bản; có thể cần đăng nhập Roboflow. Bộ này **không gán nhãn 13 class PPE**; xem lưu ý khi gộp bên dưới.

### C — Construction-PPE của Ultralytics, nguồn bổ sung tùy chọn

- [Tài liệu và mô tả class](https://docs.ultralytics.com/datasets/detect/construction-ppe) · [ZIP tải trực tiếp](https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip).
- **1.416 ảnh**, 11 class: `helmet`, `gloves`, `vest`, `boots`, `goggles`, `none`, `Person`, `no_helmet`, `no_goggle`, `no_gloves`, `no_boots`.
- Bổ sung được 5 class PPE có mặt, 4 class thiếu PPE và `person`. **Không có** `no_vest`, `mask/no_mask` hoặc hai class quần. Bỏ/kiểm tra riêng `none`, không tự suy nó thành tất cả `no_*`.
- Tài liệu nguồn ghi **AGPL-3.0** cho dataset. Kiểm tra điều khoản trước khi ghép vào sản phẩm thương mại hoặc phát hành bộ đã gộp.

## 3. Bảng bao phủ nguồn dữ liệu

| Nhóm class đích | Nguồn tải tối thiểu | Nguồn bổ sung |
|---|---|---|
| `person` | A | C (`Person`) |
| `helmet/no_helmet` | A | C |
| `vest/no_vest` | A | C chỉ có `vest` |
| `gloves/no_gloves` | A | C |
| `boots/no_boots` | A | C |
| `mask/no_mask` | A | — |
| `eye_protection/no_eye_protection` | A (`Goggles/NO-Goggles`, sau khi kiểm tra ngữ nghĩa) | C (`goggles/no_goggle`) |
| `long_pants/short_pants` | B | — |

**Tải tối thiểu A + B là có nhãn nguồn cho cả 15 class**, nhưng sự hiện diện của tên class không chứng minh mỗi class có đủ số bbox, độ phủ góc máy, hoặc chất lượng nhãn để train tốt. Chưa kiểm định ảnh và số mẫu **theo từng class** trong các file xuất.

## 4. Cách gộp trước khi train

1. Tải hai phiên bản A và B, lưu file `data.yaml` gốc của từng bộ; dùng bảng ID ở trên để **remap class ID trong từng file nhãn `.txt`**, không chỉ sửa tên trong YAML. Kiểm tra thêm định dạng tọa độ bbox và từng cặp class đối lập trên ảnh mẫu.
2. Xác định **quy tắc box nhất quán**: với `helmet/no_helmet` là vùng đầu, `mask/no_mask` là mặt, `eye_protection/no_eye_protection` là vùng mắt/mặt, `vest/no_vest` là thân trên, `gloves/no_gloves` là tay, `boots/no_boots` là chân/giày, `long_pants/short_pants` là vùng quần. Giữ hai nhãn đối lập của mỗi nhóm cùng loại vùng; nếu nguồn gán box khác nhau, chỉnh annotation trước khi trộn.
3. **Không nối mù hai bộ rồi train detector 15 class**: ở ảnh A, quần nhìn thấy nhưng chưa được đánh nhãn; ở ảnh B, PPE nhìn thấy nhưng chưa được đánh nhãn. Với loss detector YOLO thông thường, vật thể chưa gán nhãn có thể bị học thành nền. Cách an toàn là bổ sung annotation còn thiếu trên ảnh được chọn, hoặc huấn luyện hai nhánh PPE và quần riêng rồi hợp nhất kết quả theo từng `person`.
4. Những vùng không quan sát được để ở trạng thái `unknown/ignore` theo hướng dẫn gán nhãn của pipeline; riêng việc xóa bbox không đủ để tạo vùng *ignore* trong mọi trainer. Chỉ dùng `no_*` khi vùng tương ứng hiện rõ và xác nhận vi phạm. Một người trong ảnh có thể có nhiều nhãn PPE, nên cần gán box vào đúng người và theo dõi qua các frame nếu cảnh báo bằng video.
5. Giữ tập validation/test từ **camera và bối cảnh triển khai thực tế**; kiểm tra false alarm từng class, đặc biệt với bàn tay, chân và vùng mắt nhỏ. Kiểm tra ảnh trùng giữa nguồn trước khi chia lại tập để tránh rò rỉ train/test.

## 5. Ghi công và giới hạn

- Nguồn A: [Izanagi, PPE Detection and Compliance](https://universe.roboflow.com/izanagi/ppe-detection-and-compliance), CC BY 4.0.
- Nguồn B: [NCKH, Pants_v2](https://universe.roboflow.com/nckh-oufrk/pants_v2), CC BY 4.0.
- Nguồn C tùy chọn: [Ultralytics, Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe), AGPL-3.0 theo tài liệu nguồn.
- Kiểm tra lại giấy phép, nguồn gốc ảnh và điều khoản tại thời điểm tải; ghi tác giả/URL/giấy phép và thay đổi khi công bố bộ đã chuẩn hóa. Đây là danh sách **nguồn ảnh + nhãn**, không phải khẳng định có một mô hình pretrained 15 class dùng ngay.
