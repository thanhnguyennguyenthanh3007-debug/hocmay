import numpy as np

# Trọng số, đã bao gồm bias
w = np.array([1, 2, -10])

# Vector dữ liệu, đã thêm bias
x = np.array([3, 4, 1])

# Nhãn thực tế
y = -1

# Learning rate
eta = 1


# Tính w^T x
z = np.dot(w, x)

print("w^T x =", z)


# Hàm dự đoán Perceptron
if z >= 0:
    y_pred = 1
else:
    y_pred = -1

print("Nhan du doan =", y_pred)


# Kiểm tra phân lớp
if y_pred != y:
    print("Phan lop sai")

    # Perceptron update
    w = w + eta * y * x

    print("Cap nhat trong so:")
    print("w moi =", w)

else:
    print("Phan lop dung")