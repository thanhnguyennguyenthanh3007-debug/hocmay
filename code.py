# -*- coding: utf-8 -*-
"""BTL Hoc may - Phan 2: Thuc nghiem
Bai toan: du doan chu de quan tam (Sport / Entertainment / Lifestyle) cua nguoi dung Twitter
Mo hinh: Logistic Regression (softmax) va Decision Tree (ID3)
Du lieu: Dryad doi:10.5061/dryad.7sqv9s50w
"""
# ---------- 1. Khai bao thu vien ----------
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)

BASE_DIR = Path(__file__).resolve().parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "doi_10_5061_dryad_7sqv9s50w__v20240613"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE_DIR / "out"
SEED, N_SAMPLE = 42, 10000
LABELS = ["Sport", "Entertainment", "Lifestyle"]
import os; os.makedirs(OUT, exist_ok=True)

# ---------- 2. Doc du lieu tu file CSV ----------
topic = pd.read_csv(f"{DATA}/Final_topic_classification.csv")          # UserIDs, Class
info = pd.read_csv(f"{DATA}/UserInfo.csv")                              # thong tin tai khoan
beh = pd.read_csv(f"{DATA}/UserAnalysis_Data.csv", na_values=["N/A"])   # hanh vi truoc thi nghiem
PCT = ["tweets_media_pct_pre", "likes_media_pct_pre", "tweets_pol_pct_pre", "likes_pol_pct_pre"]
beh = beh[["UserIDs"] + PCT].rename(columns={"UserIDs": "key"})

# UserAnalysis_Data luu ID dang so thuc nen ghep bang khoa float64
topic["key"] = topic["UserIDs"].astype("float64")
df = topic.merge(info.drop(columns=["Unnamed: 0", "screen_name"]), left_on="UserIDs", right_on="user_id")
df = df.merge(beh, on="key", how="left")
n_raw = len(df)
df = df.dropna(subset=PCT).reset_index(drop=True)                       # bo mau thieu du lieu
n_complete = len(df)

# dac trung
created = pd.to_datetime(df["created_at"], utc=True)
df["account_age_days"] = (created.max() - created).dt.days
COUNT = ["listed_count", "favourites_count", "statuses_count", "friends_count", "followers_count"]
FEATURES = ["account_age_days"] + COUNT + PCT
df["label"] = df["Class"].map({"Sport": 0, "Entertainment": 1, "Lifestyle": 2})

# lay mau phan tang ~10.000 dong, giu ty le 3 nhan
sample, _ = train_test_split(df, train_size=N_SAMPLE, stratify=df["label"], random_state=SEED)
sample = sample.reset_index(drop=True)
sample[["UserIDs", "Class"] + FEATURES].to_csv(f"{OUT}/du_lieu_mau_10000.csv", index=False)

X = sample[FEATURES].copy()
y = sample["label"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

info_out = {
    "n_raw_merged": int(n_raw), "n_complete": int(n_complete), "n_sample": int(len(sample)),
    "class_full": df["Class"].value_counts().to_dict(),
    "class_sample": sample["Class"].value_counts().to_dict(),
    "n_train": int(len(ytr)), "n_test": int(len(yte)),
    "class_train": {LABELS[k]: int(v) for k, v in zip(*np.unique(ytr, return_counts=True))},
    "class_test": {LABELS[k]: int(v) for k, v in zip(*np.unique(yte, return_counts=True))},
    "features": FEATURES,
    "describe": sample[FEATURES].describe().T[["mean", "50%", "min", "max"]].round(2).to_dict("index"),
}

def evaluate(name, y_true, y_pred, store):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    store[name] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p.mean()), "recall_macro": float(r.mean()), "f1_macro": float(f.mean()),
        "per_class": {LABELS[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])} for i in range(3)},
        "cm": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }

res = {}
evaluate("Baseline (luon doan Sport)", yte, np.zeros_like(yte), res)

# ---------- 3. Logistic Regression ----------
# tien xu ly: log(1+x) cho cac bien dem, sau do chuan hoa (chi fit tren tap huan luyen)
LOGC = COUNT
def prep_lr(Xa):
    Z = Xa.copy()
    Z[LOGC] = np.log1p(Z[LOGC])
    return Z
scaler = StandardScaler().fit(prep_lr(Xtr))
Ztr, Zte = scaler.transform(prep_lr(Xtr)), scaler.transform(prep_lr(Xte))

# 3a. Dung thu vien
lr = LogisticRegression(max_iter=2000, random_state=SEED).fit(Ztr, ytr)
evaluate("Logistic Regression (thu vien)", yte, lr.predict(Zte), res)

# 3b. Dung cong thuc toan: softmax + Gradient Descent
def softmax(S):
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)

def fit_softmax_gd(Z, y, K=3, eta=0.5, epochs=1500):
    N = Z.shape[0]
    A = np.hstack([np.ones((N, 1)), Z])               # them x0 = 1 (he so chech)
    W = np.zeros((A.shape[1], K))
    Y = np.eye(K)[y]
    losses = []
    for _ in range(epochs):
        P = softmax(A @ W)                             # P(y=k|x)
        losses.append(-np.sum(Y * np.log(P + 1e-12)) / N)   # J(W)/N
        W += eta * A.T @ (Y - P) / N                   # w_k <- w_k + eta (y_ik - P_ik) x_i
    return W, losses

def predict_softmax(W, Z):
    A = np.hstack([np.ones((Z.shape[0], 1)), Z])
    return softmax(A @ W).argmax(axis=1)

W, losses = fit_softmax_gd(Ztr, ytr)
evaluate("Logistic Regression (cong thuc)", yte, predict_softmax(W, Zte), res)

# ---------- 4. Decision Tree (ID3) ----------
# 4a. Dung thu vien (entropy), chon do sau bang kiem tra cheo 5 lan tren tap huan luyen
def cv_macro_f1(make_model, Xa, ya):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    sc = []
    for a, b in skf.split(Xa, ya):
        m = make_model().fit(Xa[a], ya[a])
        sc.append(precision_recall_fscore_support(ya[b], m.predict(Xa[b]), labels=[0, 1, 2], zero_division=0)[2].mean())
    return float(np.mean(sc))

Xtr_np, Xte_np = Xtr.values, Xte.values
depth_scores = {d: cv_macro_f1(lambda d=d: DecisionTreeClassifier(criterion="entropy", max_depth=d,
                               min_samples_leaf=20, random_state=SEED), Xtr_np, ytr) for d in [2, 3, 4, 5, 6, 8]}
best_depth = max(depth_scores, key=depth_scores.get)
dt = DecisionTreeClassifier(criterion="entropy", max_depth=best_depth, min_samples_leaf=20,
                            random_state=SEED).fit(Xtr_np, ytr)
evaluate("Decision Tree (thu vien)", yte, dt.predict(Xte_np), res)

# 4b. Dung cong thuc toan: ID3 (Entropy, Information Gain) tren thuoc tinh roi rac (chia 4 khoang phan vi)
def entropy(y):
    if len(y) == 0: return 0.0
    p = np.bincount(y, minlength=3) / len(y)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())            # log tu nhien

def info_gain(col, y):
    H = entropy(y); N = len(y); Hx = 0.0
    for v in np.unique(col):
        mask = col == v
        Hx += mask.sum() / N * entropy(y[mask])      # H(A,S)
    return H - Hx                                    # G(A,S) = H(S) - H(A,S)

class ID3:
    def __init__(self, max_depth=4, min_samples=40, min_gain=1e-4):
        self.max_depth, self.min_samples, self.min_gain = max_depth, min_samples, min_gain
    def fit(self, D, y, attrs=None, depth=0):
        attrs = list(range(D.shape[1])) if attrs is None else attrs
        major = int(np.bincount(y, minlength=3).argmax())
        # dieu kien dung
        if entropy(y) == 0 or not attrs or len(y) < self.min_samples or depth >= self.max_depth:
            self.tree = ("leaf", major); return self if depth == 0 else ("leaf", major)
        gains = {a: info_gain(D[:, a], y) for a in attrs}
        a = max(gains, key=gains.get)                # A* = argmax G(A,S)
        if gains[a] < self.min_gain:
            node = ("leaf", major)
        else:
            children = {}
            for v in np.unique(D[:, a]):
                m = D[:, a] == v
                children[int(v)] = self.fit(D[m], y[m], [t for t in attrs if t != a], depth + 1)
            node = ("node", a, children, major)
        if depth == 0:
            self.tree = node; return self
        return node
    def _pred1(self, node, x):
        while node[0] == "node":
            ch = node[2]
            node = ch.get(int(x[node[1]]), ("leaf", node[3]))
        return node[1]
    def predict(self, D):
        return np.array([self._pred1(self.tree, x) for x in D])

# roi rac hoa: 4 khoang theo phan vi (tinh tren tap huan luyen)
qs = np.quantile(Xtr_np, [0.25, 0.5, 0.75], axis=0)
def discretize(A):
    return np.column_stack([np.searchsorted(np.unique(qs[:, j]), A[:, j], side="right") for j in range(A.shape[1])])
Dtr, Dte = discretize(Xtr_np), discretize(Xte_np)

id3_scores = {}
for d in [2, 3, 4, 5]:
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED); sc = []
    for a, b in skf.split(Dtr, ytr):
        m = ID3(max_depth=d).fit(Dtr[a], ytr[a])
        sc.append(precision_recall_fscore_support(ytr[b], m.predict(Dtr[b]), labels=[0, 1, 2], zero_division=0)[2].mean())
    id3_scores[d] = float(np.mean(sc))
best_id3 = max(id3_scores, key=id3_scores.get)
id3 = ID3(max_depth=best_id3).fit(Dtr, ytr)
evaluate("Decision Tree ID3 (cong thuc)", yte, id3.predict(Dte), res)


# 4c. Mo rong: bu mat can bang lop bang trong so lop (thu vien)
lr_b = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED).fit(Ztr, ytr)
evaluate("Logistic Regression (can bang lop)", yte, lr_b.predict(Zte), res)
dt_b = DecisionTreeClassifier(criterion="entropy", max_depth=best_depth, min_samples_leaf=20,
                              class_weight="balanced", random_state=SEED).fit(Xtr_np, ytr)
evaluate("Decision Tree (can bang lop)", yte, dt_b.predict(Xte_np), res)

# do quan trong cua thuoc tinh o goc ID3
root_gain = {FEATURES[a]: round(info_gain(Dtr[:, a], ytr), 5) for a in range(len(FEATURES))}

# ---------- 5. Truc quan hoa ----------
plt.rcParams.update({"font.size": 11})
# 5a. phan bo nhan
fig, ax = plt.subplots(figsize=(5.2, 3.4))
order = ["Sport", "Entertainment", "Lifestyle"]
vals = [info_out["class_sample"][k] for k in order]
b = ax.bar(order, vals, color=["#2f6fb0", "#e08a2c", "#4a9a5b"])
for r_, v in zip(b, vals): ax.text(r_.get_x() + r_.get_width() / 2, v + 60, str(v), ha="center")
ax.set_ylabel("Số người dùng"); ax.set_title("Phân bố 3 nhãn trong 10.000 mẫu")
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(f"{OUT}/hinh_phan_bo_nhan.png", dpi=200)

# 5b. ma tran nham lan
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
for ax, name in zip(axes, ["Logistic Regression (thu vien)", "Decision Tree (thu vien)"]):
    cm = np.array(res[name]["cm"]); cmn = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]*100:.0f}%)", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.55 else "black", fontsize=9)
    ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(LABELS, fontsize=9); ax.set_yticklabels(LABELS, fontsize=9)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế"); ax.set_title(name.split(" (")[0], fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/hinh_ma_tran_nham_lan.png", dpi=200)

# 5b2. ma tran nham lan cua cac mo hinh can bang lop
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
for ax, name in zip(axes, ["Logistic Regression (can bang lop)", "Decision Tree (can bang lop)"]):
    cm = np.array(res[name]["cm"]); cmn = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]*100:.0f}%)", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.55 else "black", fontsize=9)
    ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(LABELS, fontsize=9); ax.set_yticklabels(LABELS, fontsize=9)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế"); ax.set_title(name.split(" (")[0] + " (cân bằng lớp)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/hinh_ma_tran_can_bang.png", dpi=200)

# 5c. so sanh chi so
names = ["Logistic Regression (thu vien)", "Logistic Regression (cong thuc)", "Decision Tree (thu vien)", "Decision Tree ID3 (cong thuc)", "Logistic Regression (can bang lop)", "Decision Tree (can bang lop)"]
short = ["LR\n(thư viện)", "LR\n(công thức)", "DT\n(thư viện)", "ID3\n(công thức)", "LR\ncân bằng lớp", "DT\ncân bằng lớp"]
metrics = [("accuracy", "Accuracy"), ("precision_macro", "Precision"), ("recall_macro", "Recall"), ("f1_macro", "F1")]
fig, ax = plt.subplots(figsize=(10.5, 4.0)); w = 0.2
for i, (k, lab) in enumerate(metrics):
    ax.bar(np.arange(6) + (i - 1.5) * w, [res[n][k] for n in names], w, label=lab)
ax.axhline(res["Baseline (luon doan Sport)"]["accuracy"], color="gray", ls="--", lw=1)
ax.text(5.45, res["Baseline (luon doan Sport)"]["accuracy"] + 0.01, "baseline: luôn đoán Sport", ha="right", color="gray", fontsize=9)
ax.set_xticks(range(6)); ax.set_xticklabels(short, fontsize=9); ax.set_ylim(0, 0.8); ax.legend(ncol=4, loc="upper center", fontsize=9, frameon=False)
ax.set_title("So sánh các mô hình trên tập kiểm tra (Precision/Recall/F1: trung bình macro)", fontsize=10)
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(f"{OUT}/hinh_so_sanh.png", dpi=200)

# 5d. cay quyet dinh (thu vien) - ve 3 tang dau
fig, ax = plt.subplots(figsize=(10, 4.6))
plot_tree(dt, max_depth=2, feature_names=FEATURES, class_names=LABELS, filled=True, rounded=True, fontsize=9, ax=ax, impurity=False)
fig.tight_layout(); fig.savefig(f"{OUT}/hinh_cay_quyet_dinh.png", dpi=200)

# 5e. ham mat mat
fig, ax = plt.subplots(figsize=(5.2, 3.4)); ax.plot(losses, color="#2f6fb0")
ax.set_xlabel("Vòng lặp"); ax.set_ylabel("J(W) / N"); ax.set_title("Hàm mất mát khi huấn luyện (Gradient Descent)", fontsize=10)
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(f"{OUT}/hinh_ham_mat_mat.png", dpi=200)

info_out.update({
    "results": res, "best_depth_dt": int(best_depth), "depth_cv_dt": {str(k): round(v, 4) for k, v in depth_scores.items()},
    "best_depth_id3": int(best_id3), "depth_cv_id3": {str(k): round(v, 4) for k, v in id3_scores.items()},
    "loss_first": float(losses[0]), "loss_last": float(losses[-1]),
    "root_gain": root_gain,
    "dt_feature_importance": {f: round(float(i), 4) for f, i in zip(FEATURES, dt.feature_importances_)},
})
json.dump(info_out, open(f"{OUT}/ket_qua.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: info_out[k] for k in ["n_raw_merged", "n_complete", "n_sample", "n_train", "n_test", "best_depth_dt", "best_depth_id3", "depth_cv_dt", "depth_cv_id3", "loss_first", "loss_last"]}, indent=1))
for n, v in res.items():
    print(f"{n:38s} acc={v['accuracy']:.4f} P={v['precision_macro']:.4f} R={v['recall_macro']:.4f} F1={v['f1_macro']:.4f}")
print("root gain:", sorted(root_gain.items(), key=lambda x: -x[1])[:4])
plt.show()
