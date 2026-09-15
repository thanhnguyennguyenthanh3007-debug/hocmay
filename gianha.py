import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

np.random.seed(42)

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

for name, model in candidates.items():
    # Đo điểm MSE qua 5 Folds
    neg_mse_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)
    avg_val_mse = -np.mean(neg_mse_scores)
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