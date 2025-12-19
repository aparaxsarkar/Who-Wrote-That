import os
import json
import warnings
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

from scipy.sparse import load_npz, hstack, csr_matrix
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
import lightgbm as lgb

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=FutureWarning, module="shap")


RUN_DIR = "runs/run_20251215_201857"  
CACHE_DIR = os.path.join("runs", "cache")


FORCE_TFIDF_KEY = "ff28b1ef9777" 

N_JOBS = 4
RANDOM_STATE = 5

SHAP_SAMPLE_SIZE = 1000    
SHAP_TOPK = 20            


LGBM_V3_PARAMS = dict(
    objective="multiclass",
    n_estimators=650,
    learning_rate=0.05,
    num_leaves=127,
    min_child_samples=30,
    subsample=0.85,
    colsample_bytree=0.85,
)


idx_train = np.load(os.path.join(RUN_DIR, "idx_train.npy"))
idx_test  = np.load(os.path.join(RUN_DIR, "idx_test.npy"))
y_train   = np.load(os.path.join(RUN_DIR, "y_train.npy")).astype(np.int64)
y_test    = np.load(os.path.join(RUN_DIR, "y_test.npy")).astype(np.int64)

stylo_all = np.load(os.path.join(RUN_DIR, "stylo_all.npy"))


domain_test = np.load(os.path.join(RUN_DIR, "domain_test.npy"), allow_pickle=True)

with open(os.path.join(RUN_DIR, "classes.json")) as f:
    classes = json.load(f)

with open(os.path.join(RUN_DIR, "stylo_feature_names.json")) as f:
    stylo_names = json.load(f)

n_classes = len(classes)
assert len(domain_test) == len(y_test), "domain_test length != y_test length"


keys = {}
for fname in os.listdir(CACHE_DIR):
    if not fname.startswith("tfidf_"):
        continue
    if fname.endswith("_X_train.npz"):
        key = fname[len("tfidf_"):-len("_X_train.npz")]
        keys.setdefault(key, {})["train"] = fname
    elif fname.endswith("_X_test.npz"):
        key = fname[len("tfidf_"):-len("_X_test.npz")]
        keys.setdefault(key, {})["test"] = fname
    elif fname.endswith("_vectorizer.joblib"):
        key = fname[len("tfidf_"):-len("_vectorizer.joblib")]
        keys.setdefault(key, {})["vec"] = fname

valid_keys = [k for k, v in keys.items() if "train" in v and "test" in v and "vec" in v]
if not valid_keys:
    raise RuntimeError("No complete TF-IDF cache triplet found in runs/cache.")

if FORCE_TFIDF_KEY is not None:
    if FORCE_TFIDF_KEY not in keys or not all(x in keys[FORCE_TFIDF_KEY] for x in ["train", "test", "vec"]):
        raise RuntimeError(f"FORCE_TFIDF_KEY={FORCE_TFIDF_KEY} not found as a complete cache triplet.")
    chosen_key = FORCE_TFIDF_KEY
else:
    chosen_key = None
    for k in valid_keys:
        Xtr = load_npz(os.path.join(CACHE_DIR, keys[k]["train"]))
        if Xtr.shape[0] == len(idx_train):
            chosen_key = k
            break
    if chosen_key is None:
        chosen_key = valid_keys[0]
        print("[warn] Could not match TF-IDF cache by row count. Using first available key:", chosen_key)

train_file = keys[chosen_key]["train"]
test_file  = keys[chosen_key]["test"]
vec_file   = keys[chosen_key]["vec"]

print("[cache] Using TF-IDF cache:")
print(" ", train_file)
print(" ", test_file)

tfidf_train = load_npz(os.path.join(CACHE_DIR, train_file))
tfidf_test  = load_npz(os.path.join(CACHE_DIR, test_file))
tfidf_vec   = joblib.load(os.path.join(CACHE_DIR, vec_file))

print("[cache] TF-IDF features:", tfidf_train.shape[1])


try:
    tfidf_feature_names = list(tfidf_vec.get_feature_names_out())
except Exception:
    tfidf_feature_names = [f"tfidf_{i}" for i in range(tfidf_train.shape[1])]

feature_names = tfidf_feature_names + stylo_names


X_train = hstack([tfidf_train, csr_matrix(stylo_all[idx_train])], format="csr")
X_test  = hstack([tfidf_test,  csr_matrix(stylo_all[idx_test])],  format="csr")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


print("\n[train] Re-fitting LGBM_v3...")

lgbm_v3_model = lgb.LGBMClassifier(
    **LGBM_V3_PARAMS,
    num_class=n_classes,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    verbosity=-1
)
lgbm_v3_model.fit(X_train, y_train)

probs = lgbm_v3_model.predict_proba(X_test)   # shape (n, n_classes)
preds = np.argmax(probs, axis=1)


acc  = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, average="macro", zero_division=0)
rec  = recall_score(y_test, preds, average="macro", zero_division=0)
f1   = f1_score(y_test, preds, average="macro", zero_division=0)
cm   = confusion_matrix(y_test, preds)

# AUC (OvR)
auc_macro = None
auc_weighted = None
try:
    auc_macro = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")
    auc_weighted = roc_auc_score(y_test, probs, multi_class="ovr", average="weighted")
except Exception as e:
    print("[warn] AUC could not be computed:", e)

print("\nLGBM_v3 – FULL EVALUATION")
print(f"Accuracy:         {acc:.6f}")
print(f"Macro Precision:  {prec:.6f}")
print(f"Macro Recall:     {rec:.6f}")
print(f"Macro F1:         {f1:.6f}")
print(f"AUC Macro OvR:    {auc_macro:.6f}" if auc_macro is not None else "AUC Macro OvR:    NA")
print(f"AUC Wtd  OvR:     {auc_weighted:.6f}" if auc_weighted is not None else "AUC Wtd  OvR:     NA")

print("\nLGBM_v3 Confusion Matrix:")
print(cm)


errors = (y_test != preds)
print("\nError analysis (LGBM_v3):")
print("Number of errors:", int(errors.sum()))

err_df = pd.DataFrame({
    "true_model": [classes[i] for i in y_test[errors]],
    "pred_model": [classes[i] for i in preds[errors]],
    "domain": domain_test[errors].astype(str),
})

print("\nTop confusions:")
print(err_df.value_counts().head(10))

print("\nSHAP analysis for LGBM_v3 (stylometric-on")

#Test
rng = np.random.default_rng(RANDOM_STATE)
n = min(SHAP_SAMPLE_SIZE, X_test.shape[0])
sample_idx = rng.choice(X_test.shape[0], size=n, replace=False)


X_shap = X_test[sample_idx].toarray().astype(np.float32)


explainer = shap.TreeExplainer(lgbm_v3_model.booster_)
sv = explainer.shap_values(X_shap)

if isinstance(sv, np.ndarray) and sv.ndim == 3:
    
    sv_list = [sv[:, :, i] for i in range(sv.shape[2])]
elif isinstance(sv, list):
    sv_list = sv
else:
    raise RuntimeError(f"Unexpected SHAP output: {type(sv)} / {getattr(sv, 'shape', None)}")


S = np.stack(sv_list, axis=0)             
mean_abs = np.mean(np.abs(S), axis=(0, 1)) 

# the 59
tfidf_dim = tfidf_train.shape[1]
stylo_imp = mean_abs[tfidf_dim:]  

topk = min(SHAP_TOPK, len(stylo_names))
order = np.argsort(stylo_imp)[::-1][:topk]

print("\nTop stylometric features by mean(|SHAP|):")
for j in order:
    print(f"{stylo_names[j]:35s}  {stylo_imp[j]:.6f}")


names = [stylo_names[j] for j in order][::-1]
vals  = [stylo_imp[j]   for j in order][::-1]

plt.figure(figsize=(10, 6))
plt.barh(names, vals)
plt.title("Feature Importance — Stylometric Features")
plt.xlabel("mean shap across classes & samples")
plt.tight_layout()

out_path = os.path.join(RUN_DIR, "shap_stylo_bar.png")
plt.savefig(out_path, dpi=300)
plt.show()

print("\nSaved:", out_path)
print("\nDONE.")