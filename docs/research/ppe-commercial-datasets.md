# Nguồn dataset PPE cho sản phẩm thương mại

Ngày rà soát: 2026-09-01. Phạm vi: object detection cho `helmet`, `vest`, `gloves`, `boots`, `no_helmet`, `no_vest`, `no_gloves`, `no_boots`, gồm ảnh toàn cảnh/CCTV và person crop. Chỉ dùng nguồn do tác giả, nhà phát hành hoặc nền tảng đang phân phối dataset công bố; không tải archive để suy đoán nội dung.

> Đây là rà soát kỹ thuật về license, không phải tư vấn pháp lý. Một license mở không tự giải quyết quyền hình ảnh, quyền riêng tư, nhãn hiệu hoặc quyền của người xuất hiện; chính Creative Commons cũng nêu rõ các quyền đó có thể cần xin thêm phép ([CC BY 4.0, “Notices”](https://creativecommons.org/licenses/by/4.0/)).

## Kết luận hành động

Không có nguồn công khai nào đã kiểm chứng đồng thời thỏa cả bốn điều kiện: đủ 8 lớp, dùng thương mại rõ ràng, provenance từng ảnh rõ ràng, và phủ cả CCTV lẫn person crop.

Nên triển khai theo thứ tự sau:

1. **Nhập có điều kiện** `Dataset of Personal Protective Equipment (PPE)` của Huang & Cheng và `Construction Site Safety` của Roboflow: giữ nguyên attribution, source ID và checksum cho từng ảnh; chạy kiểm tra trùng lặp trước khi chia train/val/test.
2. **Chỉ dùng Open Images sau khi xác minh license từng ảnh**; dùng để tăng độ đa dạng của các lớp dương và `person`, không tự đổi ảnh không có box PPE thành `no_*`.
3. **Cách ly Ultralytics Construction-PPE khỏi pipeline thương mại đóng** cho tới khi pháp chế chấp thuận AGPL và provenance ảnh, hoặc Ultralytics cấp điều khoản thương mại riêng bằng văn bản.
4. **Không dùng SH17, SHWD, Pictor-PPE và các bộ Roboflow không có provenance** trong bản thương mại hiện tại.
5. Phần thiếu lớn nhất là `no_gloves`, `no_boots` và person crop. Cách ít rủi ro nhất là thu thập dữ liệu tự sở hữu tại camera triển khai, gán nhãn theo quy tắc “chỉ gán `no_*` khi vùng cơ thể tương ứng nhìn thấy đủ rõ”, rồi sinh person crop từ cùng annotation. Không cần tìm một dataset crop riêng.

CC BY 4.0 cho phép sao chép, phân phối và biến đổi cho mọi mục đích, kể cả thương mại, với điều kiện ghi công, dẫn license và ghi rõ thay đổi ([Creative Commons](https://creativecommons.org/licenses/by/4.0/)). “Được dùng thương mại” dưới đây luôn giả định các nghĩa vụ đó được thực hiện.

## Ma trận nguồn

| Nguồn | License do nguồn công bố | Quy mô / format / lớp hữu ích | Mức phù hợp | Quyết định |
|---|---|---|---|---|
| [Dataset of Personal Protective Equipment (PPE), v6](https://data.mendeley.com/datasets/zkzghjvpn2/6) | CC BY 4.0 | 2.286 ảnh 640×640; YOLO; `Helmet`, `NoHelmet`, `NoVest`, `Vest` theo bài báo dữ liệu của chính tác giả ([Data in Brief](https://pmc.ncbi.nlm.nih.gov/articles/PMC12398887/)) | Ảnh công trường/toàn cảnh; bổ sung trực tiếp 4/8 lớp | **Nhập có điều kiện** |
| [Construction Site Safety](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety) | CC BY 4.0 | Project card có `Hardhat`, `Safety Vest`, `Gloves`, `NO-Hardhat`, `NO-Safety Vest`, `Person`; bản v28 có 2.801 ảnh sau augmentation từ project 717 ảnh và xuất được YOLO/COCO/VOC/TFRecord ([v28](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/28)) | Tốt cho CCTV/full-frame; không có `no_gloves`, boots | **Nhập có điều kiện**, coi 717 ảnh gốc là đơn vị đa dạng |
| [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) | Annotation CC BY 4.0; ảnh được liệt kê CC BY 2.0 | 15.851.536 box trên 600 lớp; có thể lọc lớp dương như person/helmet/glove qua metadata và downloader chính thức ([download V7](https://storage.googleapis.com/openimages/web/download_v7.html)) | Tăng domain và hard positive toàn cảnh; không cung cấp nhãn thiếu-PPE tương đương | **Chỉ nhập ảnh đã xác minh license từng file** |
| [Ultralytics Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe/) | Ultralytics tuyên bố dataset là AGPL-3.0 | 1.416 ảnh, 11.521 box, YOLO; `helmet`, `gloves`, `vest`, `boots`, `no_helmet`, `no_gloves`, `no_boots` nhưng không có `no_vest` | Khớp 7/8 lớp, giá trị cao cho lớp boots hiếm | **Cách ly / cần pháp chế hoặc license thương mại riêng** |
| [Personal Protective Equipment Detection Dataset (5-Class)](https://data.mendeley.com/datasets/8vf7z6v5sb/1) | CC BY 4.0 | Card xác nhận object detection, góc cao, cận cảnh, che khuất, bóng/chói/bụi; card không công bố tên 5 lớp, số ảnh hay format | Có vẻ phù hợp CCTV nhưng chưa đủ metadata để mapping | **Giữ danh sách chờ**, chỉ duyệt sau audit manifest/ảnh |
| [SH17](https://github.com/ahmadmughees/SH17dataset) | CC BY-NC-SA 4.0; tác giả còn ghi mục đích giáo dục/nghiên cứu/phân tích | 8.099 ảnh, 75.994 instance, 17 lớp; có helmet, gloves, shoes, safety-vest và body parts; ảnh độ phân giải cao từ Pexels | Rất tốt về lớp dương và bộ phận nhìn thấy | **Loại khỏi thương mại** vì điều khoản NC |
| [SHWD](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset) | Repo hiển thị MIT | 7.581 ảnh; Pascal VOC; `hat` và đầu/người không đội; README nói ảnh dương lấy từ Google/Baidu và một phần âm từ SCUT-HEAD | Hữu ích cho `helmet`/`no_helmet` | **Không duyệt**: MIT của repo không chứng minh quyền ảnh thu thập từ web |
| [Pictor-PPE](https://github.com/ciber-lab/pictor-ppe) | Không thấy license dataset trong repo chính chủ | 774 ảnh crowd-sourced + 698 ảnh web-mined, 4.726 worker instance; worker/hat/vest và các trạng thái W/WH/WV/WHV | Có cả toàn cảnh và person-crop theo ba approach của tác giả | **Không duyệt** tới khi chủ sở hữu cấp license rõ ràng |
| [Dataset PPE Segmentation — WZWWorkspace](https://universe.roboflow.com/wzwworkspace/dataset-ppe-segmentation-fjqeo) | Card ghi CC BY 4.0 | 4.524 ảnh, instance segmentation; gần đủ 8 khái niệm gồm Helmet/Vest/Gloves/Shoes và các lớp No-* | Khớp lớp rất cao | **Không duyệt hiện tại**: card không nêu provenance/chain of title |
| [PPE No Personal — asimov](https://universe.roboflow.com/asimov/ppe-no-personal) | Card ghi CC BY 4.0 | 1.923 ảnh; `Person`, `Hardhat`, `Safety Vest`, `No Hardhat`, `No Safety Vest` | Bổ sung 4 lớp helmet/vest | **Không duyệt hiện tại**: card không nêu provenance |

## Đánh giá chi tiết các nguồn có thể dùng

### 1. Huang & Cheng PPE v6 — ứng viên ưu tiên

Mendeley Data, nơi tác giả phát hành bản gốc, ghi **CC BY 4.0**, 2.286 ảnh 640×640 và YOLO ([dataset card](https://data.mendeley.com/datasets/zkzghjvpn2/6)). Bài báo dữ liệu của chính tác giả nêu bốn lớp `Helmet`, `NoHelmet`, `NoVest`, `Vest`, chia 1.829 train / 457 validation và không dùng ảnh augmentation ([Data Description](https://pmc.ncbi.nlm.nih.gov/articles/PMC12398887/)).

Rủi ro không bằng 0: đây là bộ tổng hợp gồm 1.646 ảnh GDUT-HWD, 134 ảnh Roboflow, 306 ảnh Kaggle và 200 ảnh tự chụp; tác giả công bố license nguồn lần lượt Apache-2.0/CC BY 4.0/Apache-2.0 nhưng vẫn cần giữ ledger theo nguồn để attribution và xử lý yêu cầu gỡ ảnh ([bảng nguồn của tác giả](https://pmc.ncbi.nlm.nih.gov/articles/PMC12398887/)). Không coi random split có sẵn là test thương mại độc lập; chia lại theo source/origin sau dedup.

Mapping chấp nhận: `Helmet → helmet`, `Vest → vest`, `NoHelmet → no_helmet`, `NoVest → no_vest`. Trước khi nhập, lấy mẫu kiểm tra rằng box `NoHelmet`/`NoVest` bám vùng PPE hay toàn thân; không trộn hai ontology box khác nhau dưới cùng tên lớp.

### 2. Roboflow Construction Site Safety — bổ sung CCTV

Project do Roboflow Universe Projects phát hành ghi CC BY 4.0 và liệt kê các lớp PPE liên quan `Hardhat`, `Gloves`, `Safety Vest`, `NO-Hardhat`, `NO-Safety Vest`, `Person` ([project card](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety)). Bản v28 cung cấp nhiều format và ghi 2.801 ảnh, nhưng cũng cho biết tạo 5 output trên mỗi training example bằng augmentation; con số này không phải 2.801 bối cảnh độc lập ([version card](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/28)).

Chỉ giữ một lineage của mỗi ảnh gốc trước khi split. Mapping: `Hardhat → helmet`, `Safety Vest → vest`, `Gloves → gloves`, `NO-Hardhat → no_helmet`, `NO-Safety Vest → no_vest`. Không suy ra `no_gloves` từ người không có box `Gloves`.

### 3. Open Images — chỉ là nguồn tăng độ đa dạng

Google công bố V7 có 15,85 triệu bounding box trên 600 lớp và cung cấp metadata, class names, box CSV cùng downloader theo image ID ([trang chính](https://storage.googleapis.com/openimages/web/index.html), [download](https://storage.googleapis.com/openimages/web/download_v7.html)). Annotation là CC BY 4.0; ảnh được liệt kê CC BY 2.0, nhưng Google nói rõ họ **không bảo đảm** trạng thái license của từng ảnh và yêu cầu người dùng tự xác minh ([license notice](https://storage.googleapis.com/openimages/web/factsfigures_v4.html#licenses)).

Vì vậy chỉ dùng ảnh mà metadata license, URL gốc và tác giả còn xác minh được tại thời điểm ingest. Open Images thích hợp cho lớp dương và `person`; absence của box không phải bằng chứng cho `no_*` vì coverage annotation không tương đương quy tắc PPE của dự án.

## Ultralytics Construction-PPE và AGPL-3.0

Không đúng nếu nói AGPL chỉ nằm trên code hoặc YAML. Trang dataset chính thức tuyên bố rõ **Construction-PPE được phát hành dưới AGPL-3.0**, BibTeX cũng ghi `license = {AGPL-3.0}` ([License and Attribution](https://docs.ultralytics.com/datasets/detect/construction-ppe/#license-and-attribution)). File cấu hình chính thức định nghĩa 1.132 train, 143 validation, 141 test, 11 lớp và trỏ trực tiếp tới archive; header của YAML cũng là AGPL-3.0 ([raw YAML](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/construction-ppe.yaml)). Kho assets phân phối archive cũng dùng AGPL-3.0 ([assets LICENSE](https://raw.githubusercontent.com/ultralytics/assets/main/LICENSE)).

Tuy nhiên, các nguồn chính thức đã kiểm tra **không cung cấp LICENSE/NOTICE độc lập ở cấp archive hoặc provenance/license từng ảnh**. Do không tải archive theo phạm vi nghiên cứu, báo cáo không khẳng định nội dung bên trong có thêm điều khoản hay không. Kết luận chính xác là: *Ultralytics tuyên bố dataset chịu AGPL-3.0; quyền/provenance từng ảnh chưa được metadata chính thức chứng minh độc lập.*

Ultralytics nói license này hỗ trợ cả ứng dụng thương mại với attribution ([official dataset page](https://docs.ultralytics.com/datasets/detect/construction-ppe/#license-and-attribution)), nhưng “commercial” không đồng nghĩa “dùng tự do trong sản phẩm đóng”. Với mô hình proprietary, cần pháp chế quyết định nghĩa vụ copyleft đối với dataset, pipeline, checkpoint và dịch vụ mạng; không tự suy đoán checkpoint huấn luyện là hay không là derivative work. Lựa chọn vận hành an toàn là cách ly nguồn này, hoặc xin điều khoản thương mại riêng bằng văn bản.

## Nguồn bị loại hoặc giữ chờ

### SH17

Repo tác giả ghi 8.099 ảnh, 75.994 instance và 17 lớp gồm `Gloves`, `Shoes`, `Safety-vest`, `Helmet` cùng `Hands`, `Foot`, `Head`, `Person` ([classes và statistics](https://github.com/ahmadmughees/SH17dataset#classes)). Nhưng license là **CC BY-NC-SA 4.0** và disclaimer giới hạn mục đích vào giáo dục, nghiên cứu, phân tích ([license](https://github.com/ahmadmughees/SH17dataset#license)); vì vậy không phù hợp sản phẩm thương mại. Pexels cho phép dùng và sửa ảnh nhưng cấm một số cách dùng/redistribution và bảo vệ người nhận diện được ([Pexels License](https://www.pexels.com/license/)); quyền ảnh gốc không xóa điều khoản NC của annotation/dataset SH17.

### SHWD

README chính chủ nói ảnh positive lấy từ Google/Baidu, một phần negative từ SCUT-HEAD, dữ liệu Pascal VOC và hai nhãn `hat`/`person` ([SHWD README](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)). Badge MIT ở repo không đủ chứng minh tác giả có quyền cấp phép thương mại cho mọi ảnh web-scraped; không nhập nếu chưa có chain-of-title từng ảnh.

### Pictor-PPE

Repo chính chủ nêu 774 ảnh crowd-sourced, 698 ảnh web-mined và 4.726 worker instance; các approach có worker/hat/vest và trạng thái W/WH/WV/WHV, trong đó approach 3 tạo person crop ([official repository](https://github.com/ciber-lab/pictor-ppe#dataset)). Repo không có file license và Open Construction ghi license “Unspecified” ([model card](https://www.openconstruction.org/models/details.html?id=Pictor-v3)). “Publicly available” không phải quyền sử dụng thương mại; không nhập.

### Các card CC BY nhưng không có provenance

Hai card Roboflow `Dataset PPE Segmentation` và `PPE No Personal` ghi CC BY 4.0, nhưng phần mô tả không cung cấp nguồn ảnh hay chain-of-title. License do uploader chọn là tín hiệu ban đầu, chưa đủ cho dữ liệu có người nhận diện được trong sản phẩm thương mại. Giữ chúng làm danh sách liên hệ tác giả; chỉ duyệt nếu nhận được provenance, xác nhận quyền cấp phép và manifest attribution bằng văn bản.

## Quy tắc mapping và annotation bắt buộc

1. `no_helmet`: chỉ gán khi đầu nhìn thấy đủ rõ và không đội helmet.
2. `no_vest`: chỉ gán khi phần thân trên nhìn thấy đủ rõ và không mặc vest phản quang.
3. `no_gloves`: chỉ gán cho bàn tay nhìn thấy đủ rõ; tay trong túi, sau vật cản hoặc ngoài khung là **ignore**, không phải negative.
4. `no_boots`: chỉ gán khi bàn chân/giày nhìn thấy đủ rõ; chân ngoài khung hoặc bị che là **ignore**.
5. Không suy `no_*` từ “không có box lớp dương”. Mỗi nguồn phải xác nhận annotation coverage trước khi tạo negative.
6. Giữ `source_dataset`, `source_image_id`, license, URL/tác giả, checksum, kiểu ảnh (`full_frame`/`person_crop`) và `origin_group` để attribution, dedup và group split.
7. Person crop phải được sinh **sau khi chia theo origin**; mọi crop từ cùng ảnh/video/camera nằm cùng split để tránh leakage.
8. Không map `shoes` chung thành `boots` nếu annotation không phân biệt safety boots; các ảnh đó chỉ dùng làm `no_boots`/`boots` sau review thủ công theo tiêu chuẩn nhãn dự án.

## Khoảng trống dữ liệu và kế hoạch bổ sung tối thiểu

- **CCTV/full-frame:** dùng hai nguồn CC BY được duyệt có điều kiện làm seed, sau đó ưu tiên video từ camera/site thật có giấy phép nội bộ. Lấy frame thưa theo cảnh, không đếm frame liền kề như dữ liệu độc lập.
- **Person crop:** sinh deterministic crop từ person/full-frame đã duyệt, có padding để không cắt tay/chân. Đây là derivative của cùng ảnh nên giữ cùng license và attribution.
- **`no_gloves` và `no_boots`:** nguồn thương mại có provenance rõ hiện chưa đủ. Thu thập/gán nhãn nội bộ là đường ngắn nhất; chỉ dùng Ultralytics sau quyết định pháp lý.
- **Test set:** chỉ dùng site/camera/video không xuất hiện trong train; không đưa augmentation hoặc crop cùng origin qua split khác.

## Checklist trước khi tải hoặc nhập bất kỳ nguồn nào

- Lưu snapshot dataset card, license text, DOI/version và ngày truy cập.
- Lấy manifest file trước, kiểm tra class ontology và box semantics trên mẫu nhỏ.
- Xác minh license ảnh gốc, không chỉ license code/annotation/archive.
- Tạo `ATTRIBUTION.md` và ledger từng nguồn; ghi rõ mọi remap/crop/chỉnh sửa.
- Hash exact và perceptual để loại trùng giữa các nguồn trước group split.
- Quét ảnh có mặt người, logo/nhãn hiệu và chính sách quyền riêng tư theo thị trường triển khai.
- Cách ly mọi nguồn `NC`, license không rõ, hoặc không chứng minh được quyền cấp phép.
