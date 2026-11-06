# Rà soát nguồn mở rộng dataset PPE

Ngày rà soát: 2026-09-03. Mục tiêu: bổ sung `boots` và `gloves`, đồng thời tăng độ đa dạng cho `person`, `helmet`, `vest`. Chỉ xem nguồn chính chủ hoặc kho do tác giả/nhà phát hành quản lý; không coi nhãn license trên một mirror là bằng chứng về quyền đối với ảnh gốc.

> Đây là rà soát kỹ thuật, không phải tư vấn pháp lý. Quyền hình ảnh, quyền riêng tư và nhãn hiệu có thể tồn tại độc lập với bản quyền dataset.

## Kết luận ngắn

Không tìm thấy thêm một dataset bounding-box nào vừa có số lượng boots/gloves đáng kể, vừa có provenance từng ảnh đủ rõ để nhập toàn bộ một cách tự động vào sản phẩm thương mại đóng.

Đường an toàn và có ích nhất là:

1. Nhập có điều kiện **Open Images V7** cho `Glove`, `Boot`, `Helmet`, `Person`, nhưng chỉ giữ từng ảnh có license/tác giả/URL gốc còn xác minh được. `Boot` chỉ là boot nói chung, không được tự map thành safety boot.
2. Dùng **CPPE-5** để tăng độ đa dạng cho glove sau khi loại toàn bộ ảnh Google Images và chỉ giữ ảnh Flickr có license thương mại hợp lệ trong metadata.
3. Tạo một seed pool nhỏ từ **Wikimedia Commons Work gloves / Work boots**; tự sinh box bằng model rồi lọc nghiêm. Đây không phải dataset gán nhãn sẵn và ảnh sản phẩm đơn lẻ không thay thế ảnh người đang mang PPE.
4. Chưa nhập Mendeley 5-Class, Safety_PPE, Hugging Face PPE_Detection hoặc Ultralytics Construction-PPE vào pipeline thương mại đóng nếu chưa giải quyết các điều kiện nêu dưới đây.

## Danh sách xếp hạng

| Hạng | Nguồn | Giá trị thực tế | License / quyền thương mại | Tải và format | Quyết định |
|---:|---|---|---|---|---|
| 1 | [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) | 15.851.536 box / 600 lớp; có box `Glove`, `Boot`, `Helmet`, `Person`; cảnh đa dạng | Annotation CC BY 4.0; ảnh được *liệt kê* CC BY 2.0, nhưng Google không bảo đảm trạng thái từng ảnh và yêu cầu tự xác minh ([license notice](https://storage.googleapis.com/openimages/web/factsfigures_v4.html#licenses)) | CSV normalized boxes + metadata + downloader chính thức ([download](https://storage.googleapis.com/openimages/web/download_v7.html)) | **Nhập có điều kiện từng ảnh** |
| 2 | [CPPE-5 chính chủ](https://github.com/Rishit-dagli/CPPE-Dataset) | 1.029 ảnh, 4.698 box; 1.343 glove box trong 575 ảnh; ảnh người thật, nhiều góc nhìn | Repo ghi Apache-2.0, nhưng ảnh gồm Flickr và một phần nhỏ Google Images; chỉ ảnh Flickr có trường license trong metadata ([paper](https://www.cs.toronto.edu/~rishit/projects/cppe5/paper.pdf)) | COCO-style annotation; archive ~230 MB, MD5 `f4e043f983cff94ef82ef7d57a879212`; script/package chính chủ | **Nhập phần Flickr đã kiểm chứng; loại phần Google** |
| 3 | Wikimedia Commons [Work gloves](https://commons.wikimedia.org/wiki/Category:Work_gloves) + [Work boots](https://commons.wikimedia.org/wiki/Category:Work_boots) | 125 file trực tiếp trong Work gloves và 86 file trực tiếp trong Work boots, chưa tính tiểu mục; tốt làm seed/hard positive | Commons chỉ chấp nhận nội dung cho phép thương mại và phái sinh, nhưng yêu cầu kiểm tra license và hạn chế phi bản quyền trên từng file ([reuse guide](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia/en)) | Ảnh, **không có bounding box**; lấy URL/tác giả/license qua MediaWiki `imageinfo` + `extmetadata` ([API](https://www.mediawiki.org/wiki/API:Imageinfo)) | **Seed pool có điều kiện**, cần pseudo-label và lọc |
| 4 | [Mendeley PPE Detection 5-Class](https://data.mendeley.com/datasets/8vf7z6v5sb/1) | Card nói ảnh có góc cao/cận cảnh, che khuất, bóng/chói/bụi | CC BY 4.0 trên card; chưa công bố provenance, tên 5 lớp, số ảnh hoặc format ở metadata công khai | Download All từ Mendeley; cần audit archive/manifest trước | **Cách ly chờ audit** |
| 5 | [Ultralytics Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe) | 1.416 ảnh; đúng `gloves`, `boots`, `no_gloves`, `no_boots`, `helmet`, `vest`, `Person` | Chính chủ ghi AGPL-3.0 và nói hỗ trợ thương mại với attribution; nghĩa vụ copyleft với pipeline/checkpoint đóng cần được chấp thuận riêng | YOLO; 178,4 MB; URL archive và split 1.132/143/141 công khai trên YAML/trang chính thức | **Không tự nhập vào sản phẩm đóng** |

## Nguồn bị loại khỏi lần nhập tự động

### Safety_PPE trên Roboflow

[Project card](https://universe.roboflow.com/safety-jmser/safety_ppe) ghi CC BY 4.0, khoảng 6,6 nghìn ảnh; bản model v1 dùng 5.955 ảnh và có `Helmet`, `Glove`, `Goggles`, `Person`, `Shoe`, `No_Glove`, `No_Helmet`, `No_Shoe`. Tuy nhiên uploader chỉ là “Safety”, card không nêu nguồn ảnh hay chain-of-title. Theo yêu cầu provenance, **không nhập** dù ontology hấp dẫn. Ngoài ra `Shoe` không đồng nghĩa safety boot.

### PPE_Detection trên Hugging Face

[Dataset card](https://huggingface.co/datasets/ggilabert/PPE_Detection) ghi CC BY 4.0, YOLO, 668 MB và 16.014 object: 4.418 vest, 2.006 safety shoe, 2.703 helmet, 2.693 gloves cùng mask/goggles. Card không nêu số ảnh, nguồn ảnh, tác giả gốc hoặc chain-of-title; đây là uploader cá nhân, không chứng minh họ có quyền cấp CC BY cho pixels. **Không nhập** cho tới khi có provenance manifest.

### CPPE-5 nguyên bộ

Repo chính chủ dùng Apache-2.0 và cung cấp archive, nhưng bài báo nói phần lớn ảnh đến từ Flickr và một phần nhỏ từ Google Images. Metadata Flickr lưu license/tác giả/URL; metadata Google chỉ lưu URL, kích thước và MD5, không có trường license ([mô tả thu thập trong paper](https://www.cs.toronto.edu/~rishit/projects/cppe5/paper.pdf)). Vì vậy không thể nhập nguyên archive theo license repo. Chỉ subset Flickr có license tương thích mới đủ điều kiện kỹ thuật.

### Open Images `Boot` nguyên lớp

Open Images cung cấp box cho `Boot`, không phải `Safety boot`. Dùng toàn bộ làm `boots` sẽ đưa boot thời trang, quân đội, đi tuyết… vào lớp PPE và làm giảm precision. Chỉ giữ làm pretraining/hard negative hoặc sau một bộ lọc ngữ cảnh/PPE nghiêm; tuyệt đối không tạo `no_boots` từ việc thiếu box.

### SH17

[Repo chính chủ](https://github.com/ahmadmughees/SH17dataset) có 8.099 ảnh, 75.994 instance, trong đó gloves/shoes và các body-part rất hữu ích, nhưng license là CC BY-NC-SA 4.0. Điều khoản NonCommercial loại nguồn này khỏi dataset thương mại.

## Kiểm tra license cần áp dụng khi ingest

- Chỉ chấp nhận CC0/Public Domain, CC BY, hoặc license permissive tương đương; CC BY-SA chỉ nhập khi yêu cầu ShareAlike đã được chủ dự án chấp thuận rõ ràng.
- Lưu `source_dataset`, `source_image_id`, URL gốc, tác giả, license URL/version, thời điểm kiểm tra, checksum và attribution cho từng ảnh.
- Với Open Images/CPPE-5/Commons, bỏ file nếu URL/license/tác giả không còn xác minh được hoặc license có `NC`/`ND`.
- Không map `shoe` hay `boot` chung thành `boots` PPE nếu không có bằng chứng đó là safety footwear và đang được người mang.
- Không suy `no_gloves`/`no_boots` từ absence của positive box. Các lớp `no_*` chỉ hợp lệ khi tay/chân tương ứng nhìn thấy và ontology nguồn xác nhận negative được gán tường minh.
- Dedup exact + perceptual giữa mọi nguồn trước khi group split; cùng ảnh gốc và mọi crop phải ở cùng split.

## Rủi ro trùng lặp

- Roboflow Construction Site Safety hiện tại có thể xuất hiện lại trong các dataset Roboflow/Hugging Face tổng hợp; cần đối chiếu perceptual hash, không dựa vào tên file.
- CPPE-5 thu từ Flickr/Google; Open Images cũng chủ yếu dùng ảnh Flickr, nên có rủi ro overlap ảnh gốc. So cả source URL/Flickr ID lẫn perceptual hash.
- Wikimedia Commons có thể chứa ảnh nhập từ Flickr hoặc cơ quan chính phủ; dùng URL nguồn, SHA-1 của Commons và perceptual hash để tránh cùng ảnh ở nhiều bản resize/crop.
- Person crop không phải dữ liệu độc lập; luôn kế thừa `origin_group` của full frame.

## Khuyến nghị thực thi

Lần mở rộng đầu tiên nên chỉ gồm **Open Images đã xác minh + CPPE-5 Flickr đã xác minh**, vì đây là hai nguồn có annotation sẵn và metadata đủ để xây bộ lọc tự động. Commons chỉ nên thêm sau nếu số boots/gloves hợp lệ vẫn thiếu; nó cần pseudo-label nên rủi ro nhãn cao hơn. Không nên train lại trước khi audit cho thấy có boots PPE thật và số glove tăng đáng kể.
