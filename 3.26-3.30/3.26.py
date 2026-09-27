# Hàm số
def f(x):
    return x**2 - 4*x + 5


# Đạo hàm
def df(x):
    return 2*x - 4


# Giá trị khởi tạo
x = 5

# Learning rate
eta = 0.2

# Số bước cập nhật
iterations = 4


print("Gradient Descent")
print("----------------")

for i in range(iterations):
    # Tính gradient
    gradient = df(x)

    # Cập nhật x
    x = x - eta * gradient

    # Tính giá trị hàm
    fx = f(x)

    print(f"Buoc {i + 1}: x = {x:.4f}, f(x) = {fx:.4f}")


print("\nKet qua:")
print(f"x gan cuc tieu = {x:.4f}")
print(f"f(x) = {f(x):.4f}")