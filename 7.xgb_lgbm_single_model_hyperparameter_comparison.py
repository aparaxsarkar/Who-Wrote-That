import os
import json
import gc
import numpy as np

from scipy.sparse import load_npz, hstack, csr_matrix
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from xgboost import XGBClassifier
import lightgbm as lgb


RUN_DIR = "runs/run_20251215_201857"  
CACHE_DIR = os.path.join("runs", "cache")

N_JOBS = 4
RANDOM_STATE = 5

idx_train = np.load(os.path.join(RUN_DIR, "idx_train.npy"))
idx_test  = np.load(os.path.join(RUN_DIR, "idx_test.npy"))
y_train   = np.load(os.path.join(RUN_DIR, "y_train.npy"))
y_test    = np.load(os.path.join(RUN_DIR, "y_test.npy"))

stylo_all = np.load(os.path.join(RUN_DIR, "stylo_all.npy"))

with open(os.path.join(RUN_DIR, "classes.json")) as f:
    classes = json.load(f)
n_classes = len(classes)


tfidf_train = tfidf_test = None

for f in os.listdir(CACHE_DIR):
    if f.endswith("_X_train.npz"):
        tfidf_train = load_npz(os.path.join(CACHE_DIR, f))
    if f.endswith("_X_test.npz"):
        tfidf_test = load_npz(os.path.join(CACHE_DIR, f))

assert tfidf_train is not None and tfidf_test is not None


X_train = hstack([
    tfidf_train,
    csr_matrix(stylo_all[idx_train])
], format="csr")

X_test = hstack([
    tfidf_test,
    csr_matrix(stylo_all[idx_test])
], format="csr")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


def evaluate(name, probs):
    preds = probs.argmax(axis=1)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    print(f"\n{name}")
    print(f"Accuracy: {acc:.6f}")
    print(f"Macro F1: {f1:.6f}")
    print(confusion_matrix(y_test, preds))


xgb_variants = {
    "XGB_v1": dict(n_estimators=300, max_depth=6, learning_rate=0.1),
    "XGB_v2": dict(n_estimators=450, max_depth=6, learning_rate=0.08),
    "XGB_v3": dict(n_estimators=600, max_depth=5, learning_rate=0.06),
    "XGB_v4": dict(n_estimators=800, max_depth=4, learning_rate=0.05),
}

for name, cfg in xgb_variants.items():
    print(f"\n[train] {name}")
    model = XGBClassifier(
        **cfg,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    evaluate(name, probs)
    del model
    gc.collect()


lgbm_variants = {
    "LGBM_v1": dict(n_estimators=350, learning_rate=0.08, num_leaves=63,  min_child_samples=40),
    "LGBM_v2": dict(n_estimators=450, learning_rate=0.07, num_leaves=63,  min_child_samples=40),
    "LGBM_v3": dict(n_estimators=650, learning_rate=0.05, num_leaves=127, min_child_samples=30),
    "LGBM_v4": dict(n_estimators=500, learning_rate=0.06, num_leaves=95,  min_child_samples=50),
}

for name, cfg in lgbm_variants.items():
    print(f"\n[train] {name}")
    model = lgb.LGBMClassifier(
        **cfg,
        objective="multiclass",
        num_class=n_classes,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    evaluate(name, probs)
    del model
    gc.collect()

print("\nDone.")
