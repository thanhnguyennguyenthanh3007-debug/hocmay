import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

np.random.seed(42)

OUTPUT_DIR = Path(__file__).parent / "hinh_anh"
OUTPUT_DIR.mkdir(exist_ok=True)

# ==============================================================================
# BƯỚC 1: TẠO DỮ LIỆU GIÁ NHÀ THỰC TẾ (HOẶC ĐỌC TỪ FILE EXCEL / CSV CỦA BẠN)
# ==============================================================================
# Giả lập 30 căn nhà với các đặc trưng thực tế:
N = 30
dien_tich = np.random.uniform(40, 150, N)       # 40m2 đến 150m2
so_phong = np.random.choice([1, 2, 3, 4, 5], N) # 1 đến 5 phòng ngủ
khoang_cach = np.random.uniform(1, 20, N)       # 1km đến 20km cách trung tâm

# Công thức giá nhà thực tế giả định trên thị trường:
# Giá = 0.05*Diện tích (50tr/m2) + 0.3*Số phòng (300tr/phòng) - 0.15*Khoảng cách + Nhiễu
gia_thuc = 0.05 * dien_tich + 0.3 * so_phong - 0.15 * khoang_cach + 1.2
nhiễu = np.random.normal(0, 0.25, N) # Sai số ngẫu nhiên do thương lượng, hướng nhà...
gia_nha = gia_thuc + nhiễu            # Đơn vị: Tỷ VNĐ

# Tạo bảng dữ liệu giống hệt bảng bạn đang nhìn thấy
df_nha = pd.DataFrame({
    'dien_tich_m2': np.round(dien_tich, 1),
    'so_phong': so_phong,
    'khoang_cach_km': np.round(khoang_cach, 1),
    'gia_nha_ty': np.round(gia_nha, 2)
})
df_nha.to_csv(OUTPUT_DIR / "house_price.csv", index=False, encoding="utf-8-sig")

print("=== 5 DÒNG ĐẦU TIÊN CỦA BẢNG DỮ LIỆU GIÁ NHÀ ===")
print(df_nha.head())
print("-" * 60)

# X gồm 3 cột đặc trưng: [diện tích, số phòng, khoảng cách]
# y là 1 cột mục tiêu: giá nhà
X = df_nha[['dien_tich_m2', 'so_phong', 'khoang_cach_km']]
y = df_nha['gia_nha_ty']

# ==============================================================================
# BƯỚC 2: CHIA TẬP TRAIN (70%) VÀ TẬP TEST (30%)
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Tổng số căn nhà: {N}")
print(f"Số căn nhà dùng để học (Train): {len(X_train)}")
print(f"Số căn nhà niêm phong kiểm tra (Test): {len(X_test)}\n")

# ==============================================================================
# BƯỚC 3: MÔ HÌNH BỊ OVERFITTING TRÊN BÀI TOÁN GIÁ NHÀ (Bậc đa thức quá cao)
# ==============================================================================
# Khi đưa 3 biến vào đa thức bậc 3 hoặc bậc 4, số lượng tổ hợp biến tăng vọt:
# x1^3, x1^2*x2, x1*x2*x3, ... khiến mô hình có hàng chục tham số trong khi chỉ có 21 căn nhà train!
overfit_model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=3),
    LinearRegression()
)
overfit_model.fit(X_train, y_train)

train_pred_overfit = overfit_model.predict(X_train)
test_pred_overfit = overfit_model.predict(X_test)

print("=== [MÔ HÌNH OVERFITTING (ĐA THỨC BẬC 3 TRÊN 3 BIẾN)] ===")
print(f"Train MSE: {mean_squared_error(y_train, train_pred_overfit):.4f} (Khớp cực kỳ sát tập train)")
print(f"Test MSE:  {mean_squared_error(y_test, test_pred_overfit):.4f} (BÙNG NỔ! Dự đoán nhà mới cực kỳ sai lệch)")
print("-" * 60)

# ==============================================================================
# BƯỚC 4: DÙNG K-FOLD CROSS VALIDATION (K=5) ĐỂ CHỌN MÔ HÌNH TỐI ƯU
# ==============================================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Thử nghiệm các mô hình ứng viên:
# 1. Hồi quy tuyến tính bậc 1 (Linear Regression đơn giản)
# 2. Đa thức bậc 2
# 3. Hồi quy có Regularization (Ridge Regression chống overfit)
candidates = {
    "Tuyến tính bậc 1 (Linear)": make_pipeline(StandardScaler(), PolynomialFeatures(degree=1), LinearRegression()),
    "Đa thức bậc 2 (Poly deg=2)": make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), LinearRegression()),
    "Đa thức bậc 2 + Ridge (L2)": make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), Ridge(alpha=1.0)),
    "Đa thức bậc 3 (Overfit)": make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), LinearRegression()),
}

print("=== [ĐÁNH GIÁ CÁC MÔ HÌNH BẰNG 5-FOLD CROSS VALIDATION] ===")
best_name = None
best_score = float('inf')
cv_results = {}

for name, model in candidates.items():
    # Đo điểm MSE qua 5 Folds
    neg_mse_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)
    avg_val_mse = -np.mean(neg_mse_scores)
    cv_results[name] = -neg_mse_scores
    print(f"  * {name:<26}: Validation MSE trung bình = {avg_val_mse:.4f}")
    if avg_val_mse < best_score:
        best_score = avg_val_mse
        best_name = name

print(f"\n=> K-Fold CV đã tự động chọn mô hình tốt nhất là: [{best_name}]")

# ==============================================================================
# BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH TỐI ƯU TRÊN TẬP TEST VÀ SO SÁNH y vs y^
# ==============================================================================
best_model = candidates[best_name]
best_model.fit(X_train, y_train)

y_test_pred_best = best_model.predict(X_test)
mae_test = mean_absolute_error(y_test, y_test_pred_best)

print(f"\nSai số dự đoán trung bình trên nhà mới (MAE): {mae_test:.3f} tỷ VNĐ (~{mae_test*1000:.0f} triệu)")

# BẢNG SO SÁNH THỰC TẾ y VÀ DỰ ĐOÁN y^ CHO TỪNG CĂN NHÀ TẬP TEST:
print("\n=== [BẢNG SO SÁNH GIÁ THỰC TẾ y VÀ GIÁ DỰ ĐOÁN y^ TRÊN TẬP TEST] ===")
results = X_test.copy()
results['y (Giá thực tế - Tỷ)'] = np.round(y_test.values, 2)
results['y^ (Mô hình tối ưu - Tỷ)'] = np.round(y_test_pred_best, 2)
results['Độ lệch |y - y^| (Triệu)'] = np.round(np.abs(y_test.values - y_test_pred_best) * 1000, 0)
results['y^ (Mô hình Overfit - Tỷ)'] = np.round(test_pred_overfit, 2)

print(results.to_string(index=False))

# ==============================================================================
# BƯỚC 6: LƯU CÁC HÌNH MINH HỌA THEO MÔ TÍP BÀI MẪU
# ==============================================================================
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()


# So sánh sai số dự đoán giữa các mô hình.
plt.figure(figsize=(10, 5))
names = list(cv_results.keys())
scores = [np.mean(cv_results[name]) for name in names]
plt.bar(range(len(names)), scores, color=["#3b82f6", "#10b981", "#f59e0b", "#ef4444"])
plt.xticks(range(len(names)), ["Linear", "Poly 2", "Poly 2 + Ridge", "Poly 3"], rotation=15)
plt.ylabel("MSE trung binh")
plt.title("So sanh hieu qua cac mo hinh")
save_plot("degree_comparison.png")

# Minh họa chênh lệch giữa mô hình tốt nhất và mô hình overfit.
overfit_train_mse = mean_squared_error(y_train, train_pred_overfit)
overfit_test_mse = mean_squared_error(y_test, test_pred_overfit)
best_train_mse = mean_squared_error(y_train, best_model.predict(X_train))
best_test_mse = mean_squared_error(y_test, y_test_pred_best)
plt.figure(figsize=(8, 5))
plt.bar(["Train - Overfit", "Test - Overfit", "Train - Best", "Test - Best"],
        [overfit_train_mse, overfit_test_mse, best_train_mse, best_test_mse],
        color=["#fca5a5", "#dc2626", "#93c5fd", "#2563eb"])
plt.ylabel("MSE")
plt.title("So sanh overfitting va mo hinh toi uu")
plt.xticks(rotation=15)
save_plot("overfit_comparison.png")

# So sánh giá thực tế với dự đoán trên tập kiểm tra.
comparison = pd.DataFrame({
    "Thuc te": y_test.to_numpy(),
    "Mo hinh toi uu": y_test_pred_best,
    "Mo hinh overfit": test_pred_overfit,
}).sort_values("Thuc te").reset_index(drop=True)
plt.figure(figsize=(10, 5))
plt.plot(comparison["Thuc te"], marker="o", label="Thuc te", color="#111827")
plt.plot(comparison["Mo hinh toi uu"], marker="o", label="Mo hinh toi uu", color="#2563eb")
plt.plot(comparison["Mo hinh overfit"], marker="x", label="Mo hinh overfit", color="#dc2626")
plt.xlabel("Can nha trong tap test (da sap xep)")
plt.ylabel("Gia (ty VNĐ)")
plt.title("So sanh gia thuc te va gia du doan")
plt.legend()
save_plot("final_comparison.png")

# Trực quan hóa điểm MSE ở từng fold của mô hình được chọn.
best_fold_scores = cv_results[best_name]
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(best_fold_scores) + 1), best_fold_scores, color="#0f766e")
plt.axhline(np.mean(best_fold_scores), color="#dc2626", linestyle="--", label="MSE trung binh")
plt.xlabel("Fold")
plt.ylabel("MSE")
plt.title(f"K-Fold Cross-Validation: {best_name}")
plt.legend()
save_plot("kfold_visualization.png")

# Đường cong học tập cho mô hình được chọn.
train_sizes, train_scores, validation_scores = learning_curve(
    best_model, X_train, y_train, cv=kf, scoring="neg_mean_squared_error",
    train_sizes=np.linspace(0.3, 1.0, 5)
)
plt.figure(figsize=(8, 5))
plt.plot(train_sizes, -np.mean(train_scores, axis=1), marker="o", label="Train MSE")
plt.plot(train_sizes, -np.mean(validation_scores, axis=1), marker="o", label="Validation MSE")
plt.xlabel("So luong mau huan luyen")
plt.ylabel("MSE")
plt.title("Learning curve cua mo hinh toi uu")
plt.legend()
save_plot("learning_curve.png")

print(f"\nDa luu du lieu va 5 bieu do vao thu muc: {OUTPUT_DIR}")