import pandas as pd
import math

# =========================
# 1. DỮ LIỆU
# =========================
data = [
    [25, "Độc thân", "Ở cùng bố mẹ", 7, 0],
    [40, "Đã kết hôn", "Nhà sở hữu", 18, 0],
    [35, "Từng ly hôn", "Nhà thuê", 12, 1],
    [27, "Đã kết hôn", "Ở cùng bố mẹ", 9, 1],
    [31, "Độc thân", "Nhà thuê", 6, 1],
    [36, "Đã kết hôn", "Nhà sở hữu", 8, 1],
    [48, "Độc thân", "Nhà thuê", 7, 0],
    [26, "Đã kết hôn", "Nhà thuê", 8, 1],
    [33, "Từng ly hôn", "Ở cùng bố mẹ", 5, 1],
    [29, "Độc thân", "Nhà thuê", 10, 0],
    [38, "Đã kết hôn", "Nhà sở hữu", 15, 0],
    [44, "Độc thân", "Nhà sở hữu", 14, 1],
    [42, "Đã kết hôn", "Nhà sở hữu", 10, 0],
    [28, "Độc thân", "Nhà thuê", 7, 1],
    [30, "Đã kết hôn", "Ở cùng bố mẹ", 6, 1]
]

df = pd.DataFrame(data, columns=[
    "Tuổi", "Hôn nhân", "Sở hữu BĐS",
    "Thu nhập", "Rủi ro"
])

# =========================
# 2. ENTROPY
# =========================
def entropy(y):
    n = len(y)

    if n == 0:
        return 0

    p = y.value_counts() / n

    return -sum(x * math.log2(x) for x in p)


# =========================
# 3. INFORMATION GAIN
# =========================
def information_gain(df, feature, target):
    H = entropy(df[target])

    # Thuộc tính số
    if pd.api.types.is_numeric_dtype(df[feature]):

        values = sorted(df[feature].unique())
        best_gain = -1
        best_threshold = None

        for a, b in zip(values, values[1:]):
            threshold = (a + b) / 2

            left = df[df[feature] <= threshold]
            right = df[df[feature] > threshold]

            H_after = (
                len(left) / len(df) * entropy(left[target])
                + len(right) / len(df) * entropy(right[target])
            )

            gain = H - H_after

            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold

        return best_gain, best_threshold

    # Thuộc tính phân loại
    else:

        H_after = 0

        for _, group in df.groupby(feature):
            H_after += (
                len(group) / len(df)
                * entropy(group[target])
            )

        return H - H_after, None


# =========================
# 4. TÌM THUỘC TÍNH TỐT NHẤT
# =========================
def best_feature(df, features, target):

    best_gain = -1
    best_feature = None
    best_threshold = None

    for feature in features:

        gain, threshold = information_gain(
            df, feature, target
        )

        if gain > best_gain:
            best_gain = gain
            best_feature = feature
            best_threshold = threshold

    return best_feature, best_threshold, best_gain


# =========================
# 5. XÂY DỰNG CÂY ID3
# =========================
def ID3(df, features, target):

    # Nếu tất cả mẫu cùng nhãn
    if df[target].nunique() == 1:
        return df[target].iloc[0]

    # Không còn thuộc tính
    if len(features) == 0:
        return df[target].mode()[0]

    feature, threshold, gain = best_feature(
        df, features, target
    )

    # Không thể chia tiếp
    if gain <= 0:
        return df[target].mode()[0]

    tree = {}

    # Thuộc tính số
    if threshold is not None:

        left = df[df[feature] <= threshold]
        right = df[df[feature] > threshold]

        tree[f"{feature} <= {threshold}"] = ID3(
            left, features, target
        )

        tree[f"{feature} > {threshold}"] = ID3(
            right, features, target
        )

    # Thuộc tính phân loại
    else:

        new_features = [
            f for f in features if f != feature
        ]

        for value, group in df.groupby(feature):

            tree[f"{feature} = {value}"] = ID3(
                group,
                new_features,
                target
            )

    return tree


# =========================
# 6. CHẠY ID3
# =========================
features = [
    "Tuổi",
    "Hôn nhân",
    "Sở hữu BĐS",
    "Thu nhập"
]

tree = ID3(
    df,
    features,
    "Rủi ro"
)

print("CÂY ID3:")
print(tree)

print("\nENTROPY BAN ĐẦU:")
print(round(entropy(df["Rủi ro"]), 4))

print("\nINFORMATION GAIN:")
for feature in features:
    gain, threshold = information_gain(
        df, feature, "Rủi ro"
    )

    if threshold is not None:
        print(
            feature,
            "ngưỡng =", round(threshold, 2),
            "Gain =", round(gain, 4)
        )
    else:
        print(
            feature,
            "Gain =", round(gain, 4)
        )