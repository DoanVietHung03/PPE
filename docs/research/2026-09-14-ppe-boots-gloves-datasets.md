# Dataset bổ sung cho boots/no-boots và gloves/no-gloves

Ngày rà soát: 2026-09-14. Phạm vi loại trừ các nguồn đã có trong `D-FINE/configs/ppe/sources.yml` và `dataset/ppe-5c/current/licenses.csv`. Chỉ dùng trang dự án, repository hoặc bài báo chính chủ làm bằng chứng. Thứ hạng ưu tiên chất lượng và độ phù hợp; license thương mại chỉ là thông tin tham khảo.

## Kết luận

Không có một nguồn duy nhất hoàn hảo; thứ tự nên thử là:

1. **[Factory Workers](https://universe.roboflow.com/vishesh-mehta-9fd3l/factory-workers)**: 6.473 ảnh, có đủ `glove`, `no_glove`, `shoes`, `no_shoes`; đúng ontology nhất để tải thử và audit.
2. **[SH17](https://github.com/ahmadmughees/SH17dataset)**: 8.099 ảnh công nghiệp độ phân giải cao, 75.994 box, có `Hands/Gloves/Foot/Shoes`; tốt nhất để tăng đa dạng positive và học vùng tay/chân, dù không có `no_*`.
3. **[PPEs v8](https://universe.roboflow.com/personal-protective-equipment/ppes-kaxsi/dataset/8)**: 11.978 ảnh gốc, có cặp `glove/no_glove` và `safety_shoe/nosafety_shoe`; phải lấy originals thay vì 24.924 ảnh generated.
4. **Multi-category non-PPE construction dataset**: 4.060 ảnh với `no-glove/no-shoes`, đúng negative nhất nhưng phải xin trực tiếp tác giả.
5. **Dataset APD K3** và **CV-CBI-MINING SAFETY** là nguồn dự phòng sau khi audit bốn nguồn trên.

## Xếp hạng ứng viên

| Hạng | Dataset | Nhãn hữu ích | Quy mô / cảnh | Annotation và tải | License / provenance | Quyết định |
|---:|---|---|---|---|---|---|
| 1 | [Dataset APD K3](https://universe.roboflow.com/wzwworkspace/dataset-apd-k3-helmet-vest-gloves-shoes-g92gz) | `Gloves`, `No-Gloves`, `Shoes`, `No-Shoes` | 5.531 ảnh; preview cho thấy ảnh worker nhưng card không mô tả nguồn, nên chưa thể khẳng định toàn bộ là full-frame | Object detection; tải/export qua Roboflow, cần tài khoản/API key | Card ghi CC BY 4.0, nhưng uploader `WZWWorkspace` không cung cấp chain-of-title hay nguồn từng ảnh | **Ưu tiên audit**, không auto-approve |
| 2 | [CV-CBI-MINING SAFETY](https://universe.roboflow.com/miningsafetycbi/cv-cbi-mining-safety) | `boot`, `no-boot`; phần mô tả còn nêu `Boots`, `No-Boots`, `Gloves`, `No-Gloves` nhưng model page hiện chỉ hiển thị 6 lớp, cần chốt version trước khi tải | 9.899 source images; model v2 dùng 17.858 ảnh sau xử lý; mô tả nói nhiều mỏ, góc nhìn và tư thế worker | Bounding boxes, Roboflow export; chỉ nhập source/original, không nhập bản augmented | Trang ghi MIT nhưng cũng nói dữ liệu “sourced from Roboflow Universe”; provenance gốc chưa được công bố | **Ưu tiên boot**, cách ly đến khi audit |
| 3 | [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html) | `Glove`, `Boot`, `Person`; không có `no_*` đáng tin cậy | Tập dense có khoảng 1,9 triệu ảnh; cảnh thật đa dạng nhưng `Boot` gồm cả thời trang/sinh hoạt | CSV bounding boxes normalized; Google cho tải annotation và ảnh thủ công hoặc theo subset | Annotation CC BY 4.0; ảnh được liệt kê dưới CC BY 2.0 nhưng Google yêu cầu người dùng tự kiểm tra quyền của từng ảnh | **Dùng có điều kiện**: positive/hard-negative, lọc từng ảnh và tuyệt đối không map `Boot` thẳng sang safety boot |
| 4 | [CPPE-5](https://github.com/Rishit-dagli/CPPE-Dataset) | `gloves` positive; không có `no_gloves`, không có boots | 1.029 ảnh y tế đời thực, 4.698 box; 1.343 glove box trong 575 ảnh theo bài báo | Pascal VOC, repo có tool đổi sang COCO; archive khoảng 230 MB và có MD5 | Repo Apache-2.0, nhưng pixels được thu từ Flickr và một phần Google Images; license repo không tự chứng minh quyền cho mọi ảnh | **Chỉ nhập subset Flickr có license thương mại**, giới hạn trọng số vì lệch domain công trường |
| 5 | [Multi-category non-PPE construction dataset](https://www.nature.com/articles/s41598-026-53052-y) | `no-glove`, `no-shoes` cùng `no-helmet`, `no-mask`, `no-vest`; không gán positive PPE | 4.060 ảnh full-scene; 3.197 ảnh chọn từ camera đeo người tại 5 công trường, phần còn lại từ 3 bộ helmet công khai | LabelImg, Pascal VOC XML; split 2.436/812/812; **không có link tải**, chỉ “reasonable request” | Bài báo là CC BY-NC-ND 4.0; tác giả không công bố license riêng cho dataset | **Liên hệ tác giả**, chỉ dùng khi nhận giấy phép thương mại bằng văn bản |
| 6 | [SH17](https://github.com/ahmadmughees/SH17dataset) | `Hands`, `Gloves`, `Foot`, `Shoes` giúp học vùng cơ thể; không có `no_*` | 8.099 ảnh, 75.994 instance, 17 lớp; ảnh industrial đời thực từ Pexels, độ phân giải cao | YOLO labels; tải Kaggle hoặc script URL chính thức | CC BY-NC-SA 4.0, repo còn giới hạn mục đích giáo dục/nghiên cứu/phân tích | **Loại khỏi commercial training**; có thể dùng nghiên cứu offline riêng |

## Lưu ý về hai nguồn ưu tiên

### APD K3

Trang chính chủ xác nhận 8 lớp và 5.531 ảnh: `Helmet`, `Vest`, `Gloves`, `No-Gloves`, `No-Helmet`, `No-Shoes`, `No-Vest`, `Shoes`; license được khai là CC BY 4.0. Tuy nhiên card không có mô tả dataset và tại thời điểm rà soát hiển thị 0 lượt tải. Đây là đúng ontology nhất nhưng bằng chứng provenance yếu nhất. `Shoes` chỉ được nâng thành `boots` sau khi mẫu cho thấy đó là safety/work footwear đang được mang.

### CV-CBI-MINING SAFETY

Trang model xác nhận 9.899 ảnh, các lớp hiển thị gồm `boot` và `no-boot`, và dataset v2 của model có 17.858 ảnh. Chênh lệch này nhiều khả năng do generated/augmented images, nên số ảnh không được coi là số cảnh độc lập. Phần mô tả 13 lớp không khớp danh sách 6 lớp hiện tại; phải khóa đúng version rồi đọc `data.yaml`/COCO categories trước ingest.

## Mapping tối thiểu

| Source | Target | Điều kiện |
|---|---|---|
| `Gloves`, `Glove` | `gloves` | Găng đang được mang trên tay; loại găng rời/catalog |
| `No-Gloves`, `No_Glove` | state `no_gloves` | Box phải biểu diễn bàn tay trần nhìn thấy được và gắn lại được với person |
| `boot`, `Boots`, `Safety Boots` | `boots` | Footwear bảo hộ đang được mang, không phải boot thời trang |
| `Shoes`, `Shoe`, Open Images `Boot` | quarantine | Chỉ promote sau audit thủ công |
| `no-boot`, `No-Shoes`, `No_Shoe` | state `no_boots` | Bàn chân/giày không đạt chuẩn nhìn thấy được; không suy từ absence của positive box |

Với pipeline hybrid hiện tại, giữ `no_*` làm nhãn state/calibration thay vì trộn vào 5 lớp object detector. Các mẫu `no_*` phải có trạng thái visibility (`visible`, `occluded`, `unknown`) để tay/chân bị che không trở thành negative giả.

## Checklist trước khi nhập

1. Tải đúng một version gốc; không trộn generated versions.
2. Audit ngẫu nhiên tối thiểu 100 ảnh mỗi lớp và toàn bộ ảnh có `no_boots` nếu lớp nhỏ.
3. Ghi `source_image_id`, URL, uploader/tác giả, license URL/version, checksum và thời điểm kiểm tra.
4. Loại catalog, collage, găng/giày rời và box `no_*` bao cả người.
5. Exact hash + perceptual hash với dataset hiện tại trước split; mọi crop cùng `origin_group` với full frame.
6. Báo số ảnh full-frame thật, số instance hợp lệ theo lớp và phân bố kích thước sau audit; không dùng image count trên card làm proxy.

## Nguồn đã đối chiếu

- Google Open Images công bố dense annotations trên khoảng 1,9 triệu ảnh và cung cấp tải raw annotations/images: [download V7](https://storage.googleapis.com/openimages/web/download_v7.html). Thống kê chính chủ mô tả bounding boxes cho 600 lớp và 14,6 triệu box train: [facts and figures](https://storage.googleapis.com/openimages/web/factsfigures_v4.html).
- CPPE-5 repository công bố archive, checksum, đặc tính ảnh đời thực và tool VOC-to-COCO: [official repository](https://github.com/Rishit-dagli/CPPE-Dataset); thống kê lớp chi tiết nằm trong [official paper](https://www.cs.toronto.edu/~rishit/projects/cppe5/paper.pdf).
- SH17 repository công bố size, ontology, cách tải và CC BY-NC-SA 4.0: [official repository](https://github.com/ahmadmughees/SH17dataset).
- Bài Scientific Reports công bố nguồn camera, cách gán nhãn, format, split và điều kiện cấp dữ liệu: [official article](https://www.nature.com/articles/s41598-026-53052-y).

## Hành động đề xuất

Chỉ cần xin quyền/tải thử **APD K3 v1** và **CV-CBI-MINING SAFETY source version** để audit trước. Không tải thêm CPPE-5/Open Images cho tới khi audit cho thấy lỗi hiện tại thực sự thiếu positive diversity; hai nguồn đó không giải quyết trực tiếp `no_gloves/no_boots`.
