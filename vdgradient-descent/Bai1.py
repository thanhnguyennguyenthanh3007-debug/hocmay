# Hàm số
def f(x):
    return x**2 - 2


# Đạo hàm của hàm số
def df(x):
    return 2 * x


# Gradient Descent
x = 5              # Giá trị khởi tạo
learning_rate = 0.1
iterations = 50

for i in range(iterations):
    gradient = df(x)

    # Cập nhật x
    x = x - learning_rate * gradient

    print(f"Lan lap {i + 1}: x = {x:.6f}, f(x) = {f(x):.6f}")


print("\nKet qua:")
print(f"x cuc tieu ≈ {x:.6f}")
print(f"f(x) ≈ {f(x):.6f}")