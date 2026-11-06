# Pose/keypoint model cho hybrid PPE pipeline thương mại

Ngày rà soát: 2026-09-01. Mục tiêu: ước lượng keypoint cho nhiều người trong ảnh CCTV, tái sử dụng person box từ detector D-FINE, ưu tiên runtime gọn và có thể dùng trong sản phẩm thương mại. Không tải checkpoint trong quá trình rà soát.

> Đây là rà soát kỹ thuật về license, không phải tư vấn pháp lý. License của repository code, file checkpoint/model asset và dữ liệu dùng để huấn luyện là ba lớp quyền khác nhau; license Apache/BSD của code không tự động cấp quyền cho mọi checkpoint được liên kết trong model zoo.

## Kết luận hành động

**Khuyến nghị kỹ thuật:** dùng kiến trúc **RTMPose-t hoặc RTMPose-s theo kiểu top-down**, nhận trực tiếp person box từ D-FINE. Nếu cần runtime production ít dependency, thực hiện export trong một môi trường build được khóa phiên bản, sau đó chỉ deploy artifact ONNX với ONNX Runtime.

**Điều kiện pháp lý:** không đưa checkpoint RTMPose công bố sẵn vào sản phẩm đóng chỉ dựa trên Apache-2.0 của MMPose. Các config RTMPose mang tên COCO vẫn khởi tạo backbone từ checkpoint `udp-aic-coco`; nguồn chính thức không gắn một license checkpoint tổng quát, độc lập và rõ ràng cho mọi model-zoo asset. Đường an toàn là:

1. train kiến trúc Apache-2.0 từ random initialization trên keypoint data do doanh nghiệp sở hữu hoặc đã được duyệt quyền thương mại; hoặc
2. nhận xác nhận bằng văn bản cho **đúng URL/hash checkpoint**, cả quyền redistribution/derivative và provenance training data.

Nếu cần prototype ngay bằng exact asset có license rõ nhất, **MoveNet MultiPose Lightning** là phương án dự phòng có điều kiện: trang model chính thức gắn Apache-2.0, chạy nhiều người và nhanh, nhưng giới hạn tối đa sáu người, thêm TensorFlow/TFLite runtime, và model card đặt surveillance ngoài intended use. Không coi kết quả webcam/fitness là bằng chứng chất lượng CCTV xa hoặc đông người.

Không chọn MediaPipe BlazePose cho bài toán chính vì asset được kiểm chứng là single-person và model card loại multiple people, khoảng cách trên 4 m và surveillance khỏi phạm vi. Không chọn torchvision Keypoint R-CNN làm production mặc định vì nó lặp lại detector người, rất nặng, và quyền sử dụng exact COCO weights không được BSD-3 của torchvision tự động bảo đảm.

## Ma trận quyết định

| Lựa chọn | Code | Exact model/checkpoint | Multi-person và tốc độ do nguồn công bố | Phù hợp pipeline | Quyết định |
|---|---|---|---|---|---|
| MediaPipe BlazePose GHUM 3D | [MediaPipe Apache-2.0](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE) | Model card của exact asset ghi Apache-2.0 | 33 keypoint; Lite/Full/Heavy khoảng 44/18/4 FPS trên Pixel 3 CPU và 49/40/19 FPS GPU; nhưng chỉ theo dõi một người | Runtime gọn, có tracking, nhưng sai domain CCTV nhiều người | **Loại cho pipeline chính** |
| torchvision Keypoint R-CNN R50-FPN | [torchvision BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) | Weight metadata ghi COCO_V1; torchvision cảnh báo weight có thể chịu điều khoản từ training dataset | N instance theo person detections; 59,1 M tham số, 226 MB, 137,42 GFLOPs, keypoint mAP 65,0 | Có detector Faster R-CNN riêng, không tái dùng D-FINE box; cần custom C++ ops | **Không ưu tiên** |
| RTMPose-t/s trong MMPose | [MMPose Apache-2.0](https://github.com/open-mmlab/mmpose/blob/main/LICENSE); [ngoại lệ hiện liệt kê EDPose](https://github.com/open-mmlab/mmpose/blob/main/LICENSES.md) | Không thấy license tổng quát gắn riêng cho mọi checkpoint; config COCO dùng backbone AIC+COCO | Top-down theo person box; README công bố RTMPose-m 75,8 AP COCO, trên 90 FPS i7-11700 CPU và trên 430 FPS GTX 1660 Ti | Khớp nhất: tái dùng box D-FINE; có thể export ONNX | **Khuyến nghị có điều kiện**: tự train hoặc clearance exact checkpoint |
| MoveNet MultiPose Lightning | [Exact model card: Apache-2.0](https://www.kaggle.com/models/google/movenet/tensorFlow2/multipose-lightning/1) | License gắn trực tiếp vào model version | Tối đa 6 người, 17 COCO keypoint và box, nguồn công bố trên 30 FPS trên phần lớn laptop | End-to-end multi-person nhưng thêm TensorFlow/TFLite; hard cap 6 | **Prototype dự phòng**, cần test CCTV và policy review |

Các FPS trên không thể so sánh trực tiếp: phần cứng, backend, kích thước input và việc có tính thời gian detector hay không khác nhau. Benchmark quyết định phải chạy end-to-end trên video thật, gồm D-FINE, crop/resize, pose cho mọi người, NMS/tracking và transfer CPU/GPU.

## 1. MediaPipe Pose / BlazePose

Repository MediaPipe dùng Apache-2.0 ([LICENSE chính thức](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE)). Quan trọng hơn, model card BlazePose GHUM 3D gắn Apache-2.0 trực tiếp với model asset, mô tả ba biến thể Lite/Full/Heavy lần lượt khoảng 3/6/26 MB và 33 landmark gồm wrist, ankle, heel và foot index ([official model card PDF](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf)). Đây là mức rõ ràng tốt hơn việc chỉ thấy license của code.

Tuy vậy model card nói hệ thống chỉ theo dõi một người; khi có nhiều người nó chọn một người. Multiple people, người cách camera hơn 4 m và surveillance đều ở ngoài phạm vi thiết kế/đánh giá. Dữ liệu được mô tả chủ yếu là ảnh mobile AR và fitness, không phải CCTV xa. Các số Pixel 3 nêu trong bảng là inference của pose model trên domain đánh giá đó, không phải throughput hệ thống PPE nhiều người.

API Pose Landmarker hiện có tùy chọn `num_poses` mặc định 1 và video/live-stream mode dùng tracking để tránh chạy detector ở mọi frame ([Python guide chính thức](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python)). Tùy chọn API mới không thay thế bằng chứng model-card về domain validation. Chỉ nên dùng MediaPipe nếu thiết kế được thu hẹp thành một người gần camera và khóa chính xác model bundle/version đã được duyệt.

## 2. torchvision Keypoint R-CNN

torchvision code là BSD-3-Clause, nhưng README chính thức cảnh báo pretrained weights có thể có điều khoản phát sinh từ dataset huấn luyện và người dùng phải tự xác định quyền sử dụng ([torchvision repository](https://github.com/pytorch/vision)). Model metadata chính thức ghi `KeypointRCNN_ResNet50_FPN_Weights.COCO_V1`, 59.137.258 tham số, file 226,054 MB, 137,42 GFLOPs, bbox mAP 54,6 và keypoint mAP 65,0 ([official source documentation](https://docs.pytorch.org/vision/0.25/_modules/torchvision/models/detection/keypoint_rcnn.html)). Detection module cũng được đánh dấu beta và dựa vào các custom C++ operators của torchvision.

Về kỹ thuật, model trả về keypoint cho N detected instances nên hỗ trợ nhiều người, nhưng Faster R-CNN detector tích hợp làm lặp công việc của D-FINE. Nó là baseline dễ hiểu nếu stack đã là PyTorch/torchvision, không phải lựa chọn ít compute nhất.

Về dữ liệu, điều khoản COCO công bố annotation dưới CC BY 4.0 nhưng nói COCO Consortium không sở hữu copyright ảnh; việc sử dụng ảnh phải tuân thủ Flickr terms và người dùng tự chịu trách nhiệm ([COCO Terms of Use chính thức](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm)). Vì không có license model-weight độc lập được xác minh, cần clearance exact weight hoặc tự train; BSD-3 chỉ giải quyết code.

## 3. RTMPose / MMPose

MMPose code là Apache-2.0. File license bổ sung hiện liệt kê EDPose như một trường hợp riêng, không đưa ra một license checkpoint tổng quát cho RTMPose ([MMPose LICENSES](https://github.com/open-mmlab/mmpose/blob/main/LICENSES.md)). Một maintainer trả lời câu hỏi thương mại rằng theo hiểu biết của họ model COCO có thể được dùng, đồng thời khuyên hỏi chuyên gia pháp lý; đây không phải grant license gắn với exact asset ([official issue #2106](https://github.com/open-mmlab/mmpose/issues/2106)). Một issue khác cho thấy người dùng vẫn chưa có câu trả lời rõ về quyền dùng COCO-WholeBody weights trong thương mại ([official issue #3204](https://github.com/open-mmlab/mmpose/issues/3204)). Không dùng issue comment thay cho license.

RTMPose là kiến trúc top-down: nhận person bounding boxes rồi dự đoán pose. Vì vậy nó có thể tái sử dụng box D-FINE thay vì mang theo detector thứ hai. README chính thức công bố RTMPose-m đạt 75,8 AP trên COCO, trên 90 FPS trên Intel i7-11700 và trên 430 FPS trên GTX 1660 Ti; tài liệu cũng có kết quả CrowdPose ([RTMPose official README](https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/README.md)). Các số này là benchmark pose trong thiết lập của tác giả; chi phí tăng theo số person crop và pipeline demo có thể không chạy detector ở mọi frame.

Không được suy luận checkpoint tên `..._coco...` là thuần COCO. Config RTMPose-s chính thức có `init_cfg` tải backbone `cspnext-s_udp-aic-coco`; vì vậy lineage huấn luyện bao gồm AIC ít nhất ở khâu khởi tạo ([official RTMPose-s config](https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py)). Cần audit từng config, URL và hash; tránh các weight AIC/COCO-WholeBody/HumanArt nếu quyền tương ứng chưa được duyệt.

Môi trường training/export không phải “ít dependency”: hướng dẫn chính thức yêu cầu PyTorch, MMEngine và MMCV tương thích, trong đó MMCV có compiled operators; MMDetection cần cho demo dùng detector do OpenMMLab cung cấp, không cần nếu tự cấp D-FINE boxes ([installation guide](https://github.com/open-mmlab/mmpose/blob/main/docs/en/installation.md)). Tài liệu deploy hỗ trợ RTMPose qua ONNX Runtime, TensorRT, ncnn, OpenVINO, CoreML và TorchScript ([deployment guide](https://github.com/open-mmlab/mmpose/blob/main/docs/en/user_guides/how_to_deploy.md)). Do đó nên cô lập stack MMPose/MMDeploy ở build stage và giữ production runtime nhỏ, nhưng phải lưu export recipe, opset, source commit, checkpoint hash và notices để reproducibility.

## 4. MoveNet MultiPose Lightning

Trang model do Google phát hành gắn Apache-2.0 trực tiếp vào MoveNet MultiPose Lightning version 1. Card mô tả model phát hiện tối đa sáu người, trả 17 keypoint COCO cùng bounding box và chạy trên 30 FPS trên phần lớn laptop ([official model card](https://www.kaggle.com/models/google/movenet/tensorFlow2/multipose-lightning/1)). README TensorFlow.js chính thức cũng nêu giới hạn sáu người và báo benchmark WebGL khoảng 62 FPS trên desktop i9/GTX 1070, 24 FPS trên MacBook Pro 2019 và 8 FPS trên Pixel 5; inference time không đổi theo số pose đầu ra ([official TFJS MoveNet README](https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/src/movenet/README.md)).

Google mô tả dữ liệu huấn luyện gồm COCO và một dataset nội bộ từ các frame YouTube yoga/fitness/dance được gán nhãn ([official TensorFlow blog](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html)). License exact asset rõ hơn RTMPose model zoo, nhưng provenance chi tiết của tập nội bộ không được công bố trong các nguồn đã kiểm tra. Model card đặt surveillance/identity ngoài intended use và kỳ vọng người ở khoảng cách webcam 3–6 feet. Đây là cảnh báo domain/policy cần review, không tự suy diễn thành một điều khoản cấm trong Apache-2.0.

MoveNet phù hợp proof-of-concept cảnh thưa, gần camera; không phù hợp mặc định với CCTV đông người vì hard cap sáu, domain shift và việc thêm TensorFlow/TFLite vào stack PyTorch hiện có.

## Điều kiện nghiệm thu trước production

1. **License ledger:** lưu license text, model card, URL, source commit/version, checkpoint SHA-256, training config và mọi NOTICE cho từng artifact.
2. **Provenance gate:** không duyệt weight nếu chỉ có license repository. Yêu cầu model-specific grant hoặc tự train trên data đã clearance; pháp chế xem xét quyền hình ảnh/dataset ngoài copyright code.
3. **Domain test:** test theo camera/site/video tách biệt, bucket theo số người, kích thước person box, che khuất, ngày/đêm và góc nhìn. Báo cáo PCK/OKS cho wrist/ankle riêng vì chúng quyết định `no_gloves`/`no_boots`.
4. **End-to-end latency:** đo p50/p95, VRAM/RAM và throughput với số người 1/5/10/20; không dùng FPS pose-only để sizing production.
5. **Fail-safe labeling:** keypoint confidence thấp hoặc bộ phận ngoài khung/che khuất phải thành `ignore/unknown`, không tự biến thành `no_gloves` hay `no_boots`.
6. **Runtime isolation:** nếu chọn RTMPose, export reproducibly và deploy runtime ONNX tối thiểu; so khớp output PyTorch–ONNX trên một golden set trước khi khóa artifact.

