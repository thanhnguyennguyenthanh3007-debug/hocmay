import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, lr=0.1, epochs=20):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(self.epochs):
            for xi, target in zip(X, y):

                z = np.dot(xi, self.w) + self.b
                pred = 1 if z >= 0 else 0

                if pred != target:
                    self.w += self.lr * (target - pred) * xi
                    self.b += self.lr * (target - pred)

        return self

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        return np.where(z >= 0, 1, 0)


# =========================
# DỮ LIỆU
# =========================

X = np.array([
    [0.5], [0.8], [1], [1.2], [1.5], [1.8], [2],
    [2.5], [3], [3.5], [4], [4.5], [5], [5.5]
])

y = np.array([
    0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1
])


# =========================
# HUẤN LUYỆN
# =========================

model = Perceptron()
model.fit(X, y)

print("Trọng số:", model.w)
print("Bias:", model.b)
print("Dự đoán:", model.predict(X))


# =========================
# VẼ BIỂU ĐỒ
# =========================

plt.scatter(
    X[y == 0], y[y == 0],
    color="red",
    label="Không đậu"
)

plt.scatter(
    X[y == 1], y[y == 1],
    color="blue",
    label="Đậu"
)

x = np.linspace(0, 6, 100).reshape(-1, 1)

plt.plot(
    x,
    model.predict(x),
    color="green",
    linewidth=2,
    label="Perceptron"
)

plt.xlabel("Số giờ học")
plt.ylabel("Lớp dự đoán")
plt.yticks([0, 1])

plt.legend()
plt.grid()
plt.show()