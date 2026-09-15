# Dự đoán giá nhà bằng học máy

## 1. Tổng quan về dự án

Dự án áp dụng các kỹ thuật học máy để dự đoán giá nhà dựa trên một số đặc trưng, bao gồm:

* Diện tích nhà tính bằng mét vuông.
* Số lượng phòng ngủ.
* Khoảng cách từ nhà đến trung tâm thành phố.

Dự án sử dụng bộ dữ liệu giá nhà được tạo giả lập và so sánh nhiều mô hình hồi quy khác nhau nhằm tìm ra mô hình phù hợp cho bài toán dự đoán giá nhà.

Mục đích chính của dự án là minh họa các nội dung:

* Tạo và tiền xử lý dữ liệu.
* Hồi quy đa thức.
* Hiện tượng quá khớp dữ liệu.
* Phương pháp kiểm định chéo K-Fold.
* Hồi quy Ridge.
* Đánh giá mô hình bằng MSE và MAE.

## 2. Công nghệ và thư viện sử dụng

Dự án được xây dựng bằng ngôn ngữ lập trình Python với các thư viện sau:

* **NumPy**: Dùng để tạo dữ liệu số và các giá trị ngẫu nhiên.
* **Pandas**: Dùng để tạo và xử lý bảng dữ liệu giá nhà.
* **Scikit-learn**: Dùng để xây dựng, huấn luyện, kiểm định và đánh giá các mô hình học máy.

Các thành phần chính của thư viện Scikit-learn được sử dụng:

* `train_test_split`
* `KFold`
* `cross_val_score`
* `PolynomialFeatures`
* `StandardScaler`
* `LinearRegression`
* `Ridge`
* `make_pipeline`
* `mean_squared_error`
* `mean_absolute_error`

## 3. Bộ dữ liệu

Bộ dữ liệu được tạo trực tiếp trong chương trình Python.

Dự án tạo dữ liệu cho **30 căn nhà** với các thuộc tính sau:

| Thuộc tính       | Mô tả                                                         |
| ---------------- | ------------------------------------------------------------- |
| `dien_tich_m2`   | Diện tích nhà tính bằng mét vuông                             |
| `so_phong`       | Số lượng phòng ngủ                                            |
| `khoang_cach_km` | Khoảng cách từ nhà đến trung tâm thành phố, tính bằng kilômét |
| `gia_nha_ty`     | Giá nhà tính bằng tỷ đồng Việt Nam                            |

Giá nhà được tạo dựa trên công thức giả định:

```text
Giá nhà = 0.05 × Diện tích
        + 0.3 × Số phòng
        - 0.15 × Khoảng cách
        + 1.2
        + Nhiễu ngẫu nhiên
```

Biến mục tiêu `gia_nha_ty` được biểu diễn theo đơn vị **tỷ đồng Việt Nam**.

## 4. Quy trình thực hiện dự án

Chương trình được thực hiện theo các bước chính sau:

### Bước 1: Tạo bộ dữ liệu

Chương trình tạo ra các đặc trưng của căn nhà, chẳng hạn như diện tích, số phòng ngủ và khoảng cách đến trung tâm thành phố.

Sau đó, chương trình tính toán giá nhà và lưu toàn bộ thông tin vào một bảng dữ liệu `DataFrame` của Pandas.

### Bước 2: Chia bộ dữ liệu

Bộ dữ liệu được chia thành hai phần:

* **Tập huấn luyện (Training set)**: 70% dữ liệu.
* **Tập kiểm tra (Testing set)**: 30% dữ liệu.

Tập huấn luyện được sử dụng để huấn luyện mô hình, trong khi tập kiểm tra được sử dụng để đánh giá khả năng dự đoán của mô hình trên dữ liệu chưa được sử dụng trong quá trình huấn luyện.

### Bước 3: Xây dựng mô hình có hiện tượng quá khớp

Một mô hình hồi quy đa thức bậc 3 được xây dựng bằng các thành phần:

* `StandardScaler`
* `PolynomialFeatures(degree=3)`
* `LinearRegression`

Do bộ dữ liệu có kích thước nhỏ, trong khi mô hình đa thức bậc 3 tạo ra nhiều đặc trưng mới, mô hình có thể xảy ra hiện tượng **quá khớp dữ liệu (overfitting)**.

### Bước 4: So sánh các mô hình ứng viên

Chương trình so sánh bốn mô hình:

1. Hồi quy tuyến tính bậc 1.
2. Hồi quy đa thức bậc 2.
3. Hồi quy đa thức bậc 2 kết hợp với Ridge.
4. Hồi quy đa thức bậc 3.

Các mô hình được đánh giá bằng phương pháp **kiểm định chéo 5 phần (5-Fold Cross-Validation)**.

### Bước 5: Lựa chọn mô hình tốt nhất

Mô hình có giá trị MSE trung bình trên tập kiểm định thấp nhất sẽ được lựa chọn là mô hình tốt nhất.

Sau đó, mô hình được lựa chọn sẽ được huấn luyện lại bằng tập dữ liệu huấn luyện.

### Bước 6: Đánh giá mô hình cuối cùng

Mô hình tốt nhất được đánh giá trên tập kiểm tra bằng chỉ số **Sai số tuyệt đối trung bình (MAE)**.

Chương trình cũng hiển thị bảng so sánh giữa:

* Giá nhà thực tế.
* Giá nhà dự đoán bằng mô hình tốt nhất.
* Độ lệch giữa giá thực tế và giá dự đoán.
* Giá nhà dự đoán bằng mô hình bị quá khớp.

## 5. Các mô hình học máy

### Hồi quy tuyến tính

Hồi quy tuyến tính mô hình hóa mối quan hệ giữa các đặc trưng đầu vào và giá nhà bằng một hàm tuyến tính.

### Hồi quy đa thức

Hồi quy đa thức mở rộng hồi quy tuyến tính bằng cách tạo ra các tổ hợp đa thức từ các đặc trưng đầu vào.

Phương pháp này cho phép mô hình học được những mối quan hệ phức tạp hơn giữa các biến đầu vào và giá nhà.

Tuy nhiên, việc sử dụng bậc đa thức quá cao trên một bộ dữ liệu nhỏ có thể gây ra hiện tượng quá khớp.

### Hồi quy Ridge

Hồi quy Ridge bổ sung thành phần chính quy hóa L2 vào hàm mất mát nhằm hạn chế độ lớn của các hệ số trong mô hình.

Phương pháp này có thể giúp giảm hiện tượng quá khớp và cải thiện khả năng tổng quát hóa của mô hình.

## 6. Đánh giá mô hình

Dự án sử dụng hai chỉ số đánh giá chính:

### Mean Squared Error — MSE

MSE là sai số bình phương trung bình, được dùng để đo mức độ chênh lệch bình phương trung bình giữa giá trị thực tế và giá trị dự đoán.

Giá trị MSE càng thấp thì mô hình thường có khả năng dự đoán càng tốt.

### Mean Absolute Error — MAE

MAE là sai số tuyệt đối trung bình, được dùng để đo độ lệch tuyệt đối trung bình giữa giá nhà thực tế và giá nhà dự đoán.

Trong dự án này, MAE được biểu diễn theo đơn vị **tỷ đồng Việt Nam**.

## 7. Hướng dẫn chạy chương trình

### Bước 1: Cài đặt Python

Đảm bảo máy tính đã được cài đặt Python.

### Bước 2: Cài đặt các thư viện cần thiết

Mở Terminal hoặc Git Bash và chạy lệnh:

```bash
pip install numpy pandas scikit-learn
```

### Bước 3: Chạy chương trình Python

Chạy lệnh sau:

```bash
python gianha.py
```

## 8. Kết quả đầu ra dự kiến

Chương trình sẽ hiển thị:

* Năm dòng đầu tiên của bảng dữ liệu giá nhà.
* Số lượng mẫu dữ liệu dùng để huấn luyện và kiểm tra.
* MSE trên tập huấn luyện và tập kiểm tra của mô hình có nguy cơ quá khớp.
* MSE trung bình trên tập kiểm định của từng mô hình.
* Mô hình tốt nhất được lựa chọn bằng K-Fold Cross-Validation.
* MAE cuối cùng trên tập kiểm tra.
* Bảng so sánh giữa giá nhà thực tế và giá nhà dự đoán.

## 9. Hạn chế của dự án

Dự án sử dụng bộ dữ liệu giả lập thay vì dữ liệu giá nhà thực tế.

Do đó, dự án có một số hạn chế:

* Bộ dữ liệu có kích thước tương đối nhỏ.
* Giá nhà được tạo dựa trên một công thức giả định.
* Kết quả có thể không phản ánh chính xác tình hình thực tế của thị trường bất động sản.
* Mô hình chưa được kiểm tra trên một bộ dữ liệu thực tế có kích thước lớn.

## 10. Hướng phát triển trong tương lai

Một số hướng phát triển có thể thực hiện:

* Sử dụng bộ dữ liệu giá nhà thực tế.
* Tăng số lượng mẫu dữ liệu huấn luyện.
* Bổ sung thêm các đặc trưng như vị trí, tuổi của căn nhà và số tầng.
* Trực quan hóa giá nhà thực tế và giá nhà dự đoán.
* So sánh thêm các thuật toán học máy khác.
* Triển khai mô hình thành một ứng dụng web bằng Flask hoặc Streamlit.

## 11. Tác giả

Dự án được thực hiện nhằm mục đích học tập và thực hành các kiến thức về học máy, mô hình hồi quy và đánh giá mô hình bằng Python.
