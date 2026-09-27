import numpy as np

# Trọng số ban đầu
w = np.array([-2, 1, 0])

# Dữ liệu đã thêm bias
x = np.array([2, 3, 1])

# Nhãn thực tế
y = 1

# Learning rate
eta = 1

# Bước 1: tính w^T x
wx = np.dot(w, x)

print("w^T x ban đầu =", wx)

# Bước 2: dự đoán
y_pred = 1 if wx >= 0 else -1

print("Nhãn dự đoán =", y_pred)
print("Nhãn thực tế =", y)

# Kiểm tra phân lớp sai
if y_pred != y:
    print("Mẫu bị phân lớp sai!")

    # Bước 3: cập nhật Perceptron
    w = w + eta * y * x

    print("w sau cập nhật =", w)

# Bước 4: tính lại w^T x
wx_new = np.dot(w, x)

print("w^T x sau cập nhật =", wx_new)

# Dự đoán lại
y_pred_new = 1 if wx_new >= 0 else -1

print("Nhãn dự đoán sau cập nhật =", y_pred_new)