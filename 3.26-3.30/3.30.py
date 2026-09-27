import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =========================
# 1. PERCEPTRON
# =========================

class Perceptron:

    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):

        self.w = np.zeros(X.shape[1])
        self.b = 0

        for epoch in range(self.epochs):

            errors = 0

            for xi, target in zip(X, y):

                z = np.dot(xi, self.w) + self.b

                pred = 1 if z >= 0 else 0

                if pred != target:

                    self.w += self.lr * (target - pred) * xi
                    self.b += self.lr * (target - pred)

                    errors += 1

            if errors == 0:
                break

        return self

    def predict(self, X):

        z = np.dot(X, self.w) + self.b

        return np.where(z >= 0, 1, 0)


# =========================
# 2. DATASET
# =========================

data = load_breast_cancer()

X = data.data
y = data.target

print("Số mẫu:", X.shape[0])
print("Số đặc trưng:", X.shape[1])


# =========================
# 3. CHIA TRAIN / TEST
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 4. CHUẨN HÓA
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =========================
# 5. HUẤN LUYỆN
# =========================

model = Perceptron(
    lr=0.01,
    epochs=1000
)

model.fit(X_train, y_train)


# =========================
# 6. DỰ ĐOÁN
# =========================

y_pred = model.predict(X_test)


# =========================
# 7. ĐÁNH GIÁ
# =========================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== KẾT QUẢ =====")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-score :", round(f1, 4))


# =========================
# 8. VẼ BIỂU ĐỒ
# =========================

names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score"
]

values = [
    accuracy,
    precision,
    recall,
    f1
]

plt.bar(names, values)

plt.ylim(0, 1)

plt.ylabel("Score")
plt.title("Kết quả Perceptron")

plt.grid(axis="y")

plt.show()