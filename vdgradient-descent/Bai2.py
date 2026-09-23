def g(x):
    return (1/3) * x**3 - x

def gradient(x):
    return x**2 - 1


# Giá trị ban đầu
x = 2

# Learning rate
eta = 0.1

# Số vòng lặp
iterations = 100

for i in range(iterations):
    x = x - eta * gradient(x)

print("x gần cực tiểu =", x)
print("Giá trị cực tiểu =", g(x))