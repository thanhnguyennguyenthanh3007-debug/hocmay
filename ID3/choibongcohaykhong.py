import pandas as pd
import math

# =========================
# 1. DỮ LIỆU
# =========================
data = [
    ["sunny", "hot", "high", "weak", "no"],
    ["sunny", "hot", "high", "strong", "no"],
    ["overcast", "hot", "high", "weak", "yes"],
    ["rainy", "mild", "high", "weak", "yes"],
    ["rainy", "cool", "normal", "weak", "yes"],
    ["rainy", "cool", "normal", "strong", "no"],
    ["overcast", "cool", "normal", "strong", "yes"],
    ["sunny", "mild", "high", "weak", "no"],
    ["sunny", "cool", "normal", "weak", "yes"],
    ["rainy", "mild", "normal", "weak", "yes"],
    ["sunny", "mild", "normal", "strong", "yes"],
    ["overcast", "mild", "high", "strong", "yes"],
    ["overcast", "hot", "normal", "weak", "yes"],
    ["rainy", "mild", "high", "strong", "no"]
]

df = pd.DataFrame(data, columns=[
    "outlook",
    "temperature",
    "humidity",
    "wind",
    "play"
])


# =========================
# 2. TÍNH ENTROPY
# =========================
def entropy(y):
    n = len(y)

    if n == 0:
        return 0

    counts = y.value_counts()

    result = 0

    for count in counts:
        p = count / n
        result -= p * math.log(p)

    return result


# =========================
# 3. INFORMATION GAIN
# =========================
def information_gain(df, feature, target):

    H_before = entropy(df[target])

    H_after = 0

    for value, group in df.groupby(feature):

        weight = len(group) / len(df)

        H_after += weight * entropy(group[target])

    gain = H_before - H_after

    return gain


# =========================
# 4. CHỌN THUỘC TÍNH TỐT NHẤT
# =========================
def best_feature(df, features, target):

    best_gain = -1
    best = None

    for feature in features:

        gain = information_gain(
            df,
            feature,
            target
        )

        if gain > best_gain:
            best_gain = gain
            best = feature

    return best, best_gain


# =========================
# 5. THUẬT TOÁN ID3
# =========================
def ID3(df, features, target):

    # Nếu tất cả mẫu cùng class
    if df[target].nunique() == 1:
        return df[target].iloc[0]

    # Nếu hết thuộc tính
    if len(features) == 0:
        return df[target].mode()[0]

    # Tìm thuộc tính tốt nhất
    best, gain = best_feature(
        df,
        features,
        target
    )

    tree = {
        best: {}
    }

    # Các thuộc tính còn lại
    remaining_features = [
        f for f in features
        if f != best
    ]

    # Chia dữ liệu theo thuộc tính tốt nhất
    for value, group in df.groupby(best):

        tree[best][value] = ID3(
            group,
            remaining_features,
            target
        )

    return tree


# =========================
# 6. CHẠY ID3
# =========================
features = [
    "outlook",
    "temperature",
    "humidity",
    "wind"
]

tree = ID3(
    df,
    features,
    "play"
)

print("CÂY ID3:")
print(tree)


# =========================
# 7. IN ENTROPY + GAIN
# =========================
print("\nEntropy ban dau:")
print(round(entropy(df["play"]), 4))

print("\nInformation Gain:")

for feature in features:

    gain = information_gain(
        df,
        feature,
        "play"
    )

    print(
        feature,
        "=", round(gain, 4)
    )