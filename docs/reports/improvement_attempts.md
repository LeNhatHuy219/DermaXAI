# Báo cáo các hướng cải tiến ECBM đã thử nghiệm

Báo cáo này tổng kết các hướng cải tiến mô hình ECBM mà em đã thử trên bộ dữ liệu Derm7pt. Với mỗi hướng, em trình bày hạn chế mà nó nhắm tới, ý tưởng, cách làm thực nghiệm, kết quả và nhận xét. Đến thời điểm hiện tại, chưa hướng nào cải thiện được mô hình một cách rõ ràng. Vì vậy phần cuối báo cáo phân tích nguyên nhân chung và đề xuất hướng đi tiếp theo.

Các kết quả đều được lặp lại với ba seed (42, 123, 2026) và ghi dưới dạng trung bình ± độ lệch chuẩn giữa các seed. Tập test của Derm7pt đã được dùng ở các vòng đánh giá trước. Vì vậy, thử nghiệm mới nhất (mục 2.4) chỉ dùng tập train và validation, không đọc lại tập test.

## 1. Hạn chế của ECBM: động lực cải tiến

ECBM dự đoán chẩn đoán từ ba thành phần năng lượng:

- E_class đo mức phù hợp giữa ảnh và nhãn.
- E_concept đo mức phù hợp giữa ảnh và từng concept.
- E_global đo mức phù hợp giữa bộ concept và nhãn.

Khi suy luận, ba thành phần được cộng lại theo các trọng số λ. Phân tích ở báo cáo tổng kết (mục 5) cho thấy cách kết hợp này có năm hạn chế chính:

| Hạn chế                                                                                                 | Biểu hiện trên tập test                                 |
| --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Chẩn đoán gần như chỉ dựa vào nhánh ảnh E_class                                                 | 97% quyết định của ECBM trùng với riêng E_class      |
| Với trọng số λ_global = 0.01 của bài báo gốc, sửa concept không thay đổi được chẩn đoán | Sửa đủ 7 concept chỉ tăng Macro F1 thêm 0.004         |
| Concept của ECBM kém hơn CBM, và càng kém khi tăng λ_global                                       | Concept accuracy 58.7% (λ = 1) so với 66.7% của Hard CBM |
| Kết quả dao động mạnh giữa các seed                                                                | Concept accuracy với λ = 1: 62.8% / 49.7% / 63.4%         |
| E_global mạnh khi nhận concept đúng nhưng yếu khi nhận concept do mô hình dự đoán             | ROC-AUC trên validation: 0.928 so với 0.818               |

Khi chuẩn bị các thử nghiệm sau, em đo thêm và thấy nhánh ảnh E_class gần như **học thuộc tập train**:

|                       | Train      | Validation |
| --------------------- | ---------- | ---------- |
| Macro F1 của E_class | 0.98–1.00 | 0.78–0.80 |
| NLL của E_class      | 0.01–0.06 | 0.59–0.74 |

Trên ca mới, E_class vẫn rất tự tin (xác suất lớn nhất trung bình 0.91–0.94) dù chỉ đúng khoảng 78–80%. Điều này giải thích vì sao E_class lấn át phần concept khi kết hợp.

## 2. Các hướng cải tiến đã thử

### 2.1. Chọn lại trọng số λ_global

**Ý tưởng.** Với λ_global = 0.01 của bài báo gốc, phần thông tin đi qua concept quá nhỏ nên việc bác sĩ sửa concept gần như vô tác dụng. Em chọn lại λ_global cho phù hợp với Derm7pt.

**Thực nghiệm.** Em thử sáu giá trị λ_global từ 0.01 đến 3 trên tập validation, với điều kiện chẩn đoán tự động không được giảm. Kế hoạch chọn được chốt trước khi chạy. Giá trị được chọn là λ_global = 1, sau đó chấm test một lần.

**Kết quả.**

|                                                     | λ_global = 0.01 (bài báo gốc) | λ_global = 1 (đã chọn) |
| --------------------------------------------------- | --------------------------------- | -------------------------- |
| Macro F1 tăng thêm khi sửa đủ 7 concept (test) | +0.004                            | +0.048                     |
| Concept accuracy (test)                             | 63.9%                             | 58.7%                      |
| Quyết định trùng với E_class (test)            | 99.9%                             | 97.0%                      |

**Nhận xét.** Hướng này **thành công một phần** và hiện là cấu hình chính của khóa luận: sửa concept đã thay đổi được chẩn đoán. Tuy nhiên có một đánh đổi. Khi cho concept ảnh hưởng nhiều hơn tới chẩn đoán, chẩn đoán cũng kéo lệch concept hiển thị cho bác sĩ, nên concept accuracy giảm khoảng 5 điểm. Ở chế độ tự động, chẩn đoán vẫn do nhánh ảnh quyết định trong 97% trường hợp.

### 2.2. Giới hạn mức sửa concept (ECBM Guard)

**Ý tưởng.** Khi suy luận, concept có thể bị kéo xa khỏi những gì ảnh thể hiện chỉ để "hợp" với nhãn. Em thêm một cơ chế học ngân sách cho từng nhóm concept: concept chỉ được sửa so với dự đoán ban đầu từ ảnh trong một giới hạn, vượt quá thì bị phạt.

**Thực nghiệm.** Hướng này được làm trong một repo riêng (`ECBM_Derm7pt`) với quy trình hơi khác: chọn checkpoint theo accuracy, ngưỡng 0.5. Vì vậy chỉ so được với Soft ECBM và Hard ECBM chạy cùng quy trình đó.

**Kết quả.**

| Mô hình  | Macro F1 (validation) | Macro F1 (test) | Concept accuracy (test) |
| ---------- | --------------------- | --------------- | ----------------------- |
| Soft ECBM  | 0.756                 | 0.728 ± 0.024  | 69.9%                   |
| Hard ECBM  | 0.759                 | 0.712 ± 0.007  | 69.4%                   |
| ECBM Guard | 0.769                 | 0.720 ± 0.005  | 69.3%                   |

**Nhận xét.** Guard ổn định hơn giữa các seed nhưng không tốt hơn hai baseline. Concept accuracy không đổi, và trên test số concept bị sửa sai vẫn nhiều hơn số được sửa đúng. Nguyên nhân là Guard chỉ hạn chế việc concept bị kéo lệch, chứ không làm bản thân concept dự đoán từ ảnh đúng hơn.

### 2.3. Bổ sung năng lượng mật độ (ECBM+)

**Ý tưởng.** Dựa trên Guard, em thêm một thành phần năng lượng mới để đẩy những concept đang dự đoán sai về phía trạng thái đúng.

**Thực nghiệm.** Giữ nguyên Guard và chỉ huấn luyện thành phần mới. Hệ số của thành phần này được chọn trên validation của một seed rồi áp dụng cho hai seed còn lại.

**Kết quả.** Hệ số được chọn chỉ tốt trên đúng seed dùng để chọn. Trên hai seed còn lại, Macro F1 validation thấp hơn Guard (0.740 so với 0.764, và 0.773 so với 0.782). Trên test, ECBM+ đạt Macro F1 0.732 ± 0.017 so với 0.720 ± 0.005 của Guard. Mức chênh 0.012 này nhỏ hơn độ dao động giữa các seed của chính ECBM+. Số concept được sửa đúng nhiều hơn số bị sửa sai khoảng 12 ô, trên tổng số 2.765 ô concept của tập test.

**Nhận xét.** Hiệu quả quá nhỏ để coi là cải tiến. Em nhận thấy thành phần mới được học từ concept dự đoán trên chính tập train, nơi concept đúng tới 88–91%, trong khi trên validation chỉ đúng 69–70%. Mô hình học sửa lỗi trên một tập dễ hơn nhiều so với ca thật. Quan sát này dẫn tới thiết kế của hướng tiếp theo.

### 2.4. Đồng nhất cách huấn luyện với cách suy luận

**Ý tưởng.** Khi huấn luyện, mỗi thành phần năng lượng học một bài toán riêng: E_class đoán nhãn từ ảnh, E_concept đoán concept, E_global đoán nhãn từ concept. Khi suy luận, chúng lại được cộng chung và chuẩn hóa cùng lúc. Mục tiêu lúc học và cách dùng lúc chạy vì vậy không khớp nhau. Em đề xuất học lại bước kết hợp bằng đúng phân phối mà mô hình dùng khi suy luận. Bước học lại này dùng dữ liệu **ngoài mẫu**, để tránh vấn đề học thuộc nêu ở mục 1 và 2.3.

**Thực nghiệm.** Để có dữ liệu ngoài mẫu, em chia 413 ca train thành 5 phần. Mỗi lần em huấn luyện một mô hình trên 4 phần và lấy dự đoán của nó cho phần còn lại (cross-fitting). Như vậy mỗi ca train đều có một dự đoán từ mô hình chưa từng thấy ca đó. Phần trích xuất đặc trưng từ ảnh được giữ cố định, chỉ học lại phần kết hợp. Em so sánh bốn cách:

- **Hiệu chỉnh trọng số:** chỉ học lại bốn trọng số kết hợp các thành phần.
- **Học lại E_global:** dùng mục tiêu mới.
- **Học lại E_global có mô phỏng can thiệp:** khi huấn luyện, ngẫu nhiên giữ một số concept ở giá trị đúng, như khi bác sĩ đã xác nhận. Cách này được thêm sau khi xem kết quả của hai cách đầu, nên chỉ mang tính thăm dò.
- **Mốc so sánh:** học lại E_global bằng cách cũ, cùng số bước.

Tiêu chí thành công được chốt trước khi chạy: phương pháp phải tốt hơn mốc so sánh ít nhất 0.01 về hiệu quả can thiệp, ở ít nhất 2 trên 3 seed, đồng thời không làm giảm chẩn đoán tự động và chất lượng concept. Hiệu quả can thiệp được đo bằng diện tích dưới đường Macro F1 khi lần lượt sửa từ 0 đến 7 concept.

**Kết quả trên validation.**

| Cách làm                                   | Macro F1 tự động | Concept Macro F1 | Hiệu quả can thiệp | Số seed đạt tiêu chí |
| -------------------------------------------- | ------------------- | ---------------- | --------------------- | ------------------------- |
| Mốc so sánh (cách cũ)                    | 0.777 ± 0.003      | 0.492            | 0.790 ± 0.003        | —                        |
| Hiệu chỉnh trọng số                      | 0.779 ± 0.013      | 0.510            | 0.787 ± 0.028        | 1/3                       |
| Học lại E_global                           | 0.775 ± 0.007      | 0.492            | 0.785 ± 0.005        | 0/3                       |
| Học lại E_global có mô phỏng can thiệp | 0.774 ± 0.007      | 0.490            | 0.792 ± 0.003        | 0/3                       |

**Nhận xét.**

- **Không cách nào đạt tiêu chí.** Hiệu chỉnh trọng số cho kết quả rất tốt ở seed đầu tiên (+0.026) nhưng ngược lại ở hai seed sau (−0.010 và −0.024). Nếu chỉ chạy một seed, em đã có thể kết luận sai.
- **Phát hiện quan trọng nhất** đến từ chính cách hiệu chỉnh trọng số. Ở hai trên ba seed, trọng số của E_global được học về 0. Nghĩa là khi chỉ tối ưu cho chẩn đoán tự động, mô hình tự bỏ hẳn kênh concept → nhãn, trong khi đó chính là kênh cần cho việc bác sĩ can thiệp. Kết quả này cho thấy bằng số liệu rằng không có một giá trị λ nào tốt cho cả chẩn đoán tự động lẫn can thiệp, đúng như đánh đổi đã thấy ở mục 2.1.
- **Học lại E_global làm concept dồn về các trạng thái phổ biến.** Concept accuracy tăng khoảng 6 điểm nhưng concept Macro F1 không tăng. Mô hình chủ yếu học tần suất xuất hiện cùng nhau của các concept trong 413 ca train.
- **Mô phỏng can thiệp khi huấn luyện giúp đúng hướng nhưng quá ít.** Cách này tốt hơn học lại E_global thông thường ở cả 3 seed, nhưng chỉ đưa kết quả về ngang mốc so sánh.
- **Dữ liệu ngoài mẫu là thật sự cần thiết.** Trên dữ liệu trong mẫu, mô hình gần như không còn sai số để học (khoảng 0.01), trong khi trên dữ liệu ngoài mẫu và validation sai số là 0.18–0.20. Bài học này đúng với mọi cách học thêm trên đầu ra của mô hình, kể cả ECBM+ ở mục 2.3.

## 3. Nhận xét chung: vì sao các hướng cải tiến chưa thành công

Có thể hình dung ECBM chẩn đoán từ hai nguồn thông tin:

- **Nguồn thứ nhất** đi thẳng từ ảnh tới nhãn. Nguồn này mạnh nhưng là hộp đen và quá tự tin trên ca mới.
- **Nguồn thứ hai** đi từ ảnh qua concept rồi tới nhãn. Nguồn này giải thích được, nhưng concept dự đoán sai khoảng một phần ba.

Cả sáu hướng em đã thử đều chỉ thay đổi **cách trộn hai nguồn này**, hoặc giới hạn mức concept bị sửa. Không hướng nào làm concept dự đoán từ ảnh đúng hơn. Trộn hai nguồn đều có điểm yếu thì không tạo ra thông tin mới: nghiêng về nguồn thứ nhất thì mô hình thành hộp đen, nghiêng về nguồn thứ hai thì mô hình phụ thuộc vào concept sai.

Nhiều bằng chứng cho thấy nút thắt nằm ở chất lượng concept:

- Khi nhận concept đúng, E_global đạt Macro F1 0.849. Khi nhận concept do mô hình dự đoán, con số này chỉ còn 0.762.
- Trên test, sửa đủ 7 concept nâng Macro F1 của Hard CBM từ 0.729 lên 0.793. Khoảng 0.06–0.09 Macro F1 bị mất đi do nhận diện concept kém, và không cách trộn nào lấy lại được.
- Ở mọi biến thể đã thử, kể cả trong repo phụ, concept accuracy đều chỉ nằm trong khoảng 56–71%.

Ngoài ra còn hai giới hạn đến từ dữ liệu:

- **Bộ dữ liệu có trần.** Theo Nápoles và cộng sự (Sci Rep, 2026), 30,3% số ảnh Derm7pt có bộ 7 concept giống hệt nhau nhưng chẩn đoán khác nhau, và 54% ca melanoma nằm trong nhóm này. Kể cả khi concept đúng hoàn toàn, accuracy cũng không vượt được 92,1%.
- **Dữ liệu ít nên khó thấy cải thiện nhỏ.** Tập validation chỉ có 61 ca melanoma, tập test có 101 ca. Macro F1 dao động 0.01–0.03 giữa các seed, nên các cải tiến nhỏ hơn khoảng 0.02 không phân biệt được với nhiễu.

## 4. Bài học và hướng tiếp theo

Về cách làm thực nghiệm, em rút ra ba bài học:

1. Phải lặp lại nhiều seed trước khi kết luận.
2. Nên chốt tiêu chí thành công trước khi chạy, để không vô tình chọn lại theo kết quả.
3. Mọi bước học thêm trên đầu ra của mô hình cần dùng dữ liệu ngoài mẫu.

Vì nút thắt nằm ở khâu nhận diện concept từ ảnh, các hướng có khả năng cải thiện thật sự là:

- **Tăng độ phân giải ảnh đầu vào.** Ảnh gốc khoảng 768 × 512 đang được thu về 224 px, nên các cấu trúc nhỏ như dots/globules hay streaks có thể bị mất chi tiết.
- **Dùng backbone mạnh hơn**, hoặc backbone được huấn luyện trước trên ảnh soi da.
- **Bổ sung dữ liệu có nhãn concept**, ví dụ nhãn cấu trúc của ISIC 2018 Task 2 (pigment network, streaks, globules…), để huấn luyện trước bộ nhận diện concept.

Em chưa thử các hướng này. Kể cả khi thành công, chúng vẫn bị giới hạn bởi trần của bộ dữ liệu nêu ở mục 3. Trong phạm vi khóa luận, em dự kiến giữ nguyên kết quả chính hiện có và trình bày các phân tích trên như một đóng góp: chỉ ra định lượng vì sao ECBM trên Derm7pt chưa vượt được CBM và nút thắt nằm ở đâu.
