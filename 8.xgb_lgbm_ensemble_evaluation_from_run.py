import os
import json
import gc
import numpy as np
import pandas as pd

from scipy.sparse import load_npz, hstack, csr_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier

from xgboost import XGBClassifier
import lightgbm as lgb



RUN_DIR = "runs/run_20251215_201857"   
CACHE_DIR = os.path.join("runs", "cache")

TRAIN_PATH = "raid_data/train_none.csv"  
DO_DOMAIN_ERROR_ANALYSIS = True         

N_JOBS = 4
RANDOM_STATE = 5
MODELS_DEFAULT = ["cohere-chat", "gpt4", "mistral-chat", "mpt-chat", "llama-chat"]
#stacking 
CV3 = 3
CV5 = 5


STACK_XGB_ESTIMATORS = 180
STACK_LGBM_ESTIMATORS = 180


FULL_XGB_V2 = dict(n_estimators=450, max_depth=6, learning_rate=0.08)
FULL_LGBM_V3 = dict(n_estimators=650, learning_rate=0.05, num_leaves=127, min_child_samples=30)


VOTE_WEIGHTS = {
    "VOTE-1_LGBM3_XGB2__w11": (["LGBM_v3", "XGB_v2"], [1.0, 1.0]),
    "VOTE-2_LGBM3_XGB2__w21": (["LGBM_v3", "XGB_v2"], [2.0, 1.0]),
    "VOTE-3_LGBM3_XGB2_LR__w210.5": (["LGBM_v3", "XGB_v2", "LR"], [2.0, 1.0, 0.5]),
    "VOTE-4_LGBM3_XGB2_LR_SGD__w210.50.5": (["LGBM_v3", "XGB_v2", "LR", "SGD"], [2.0, 1.0, 0.5, 0.5]),
}


idx_train = np.load(os.path.join(RUN_DIR, "idx_train.npy"))
idx_test  = np.load(os.path.join(RUN_DIR, "idx_test.npy"))
y_train   = np.load(os.path.join(RUN_DIR, "y_train.npy"))
y_test    = np.load(os.path.join(RUN_DIR, "y_test.npy"))

stylo_all = np.load(os.path.join(RUN_DIR, "stylo_all.npy"))

with open(os.path.join(RUN_DIR, "classes.json")) as f:
    classes = json.load(f)
n_classes = len(classes)


cfg_path = os.path.join(RUN_DIR, "config.json")
if os.path.exists(cfg_path):
    with open(cfg_path) as f:
        CFG = json.load(f)
    MODELS = CFG.get("MODELS", MODELS_DEFAULT)
    MAX_ROWS_AFTER_FILTER = CFG.get("MAX_ROWS_AFTER_FILTER", None)
    RANDOM_STATE = CFG.get("RANDOM_STATE", RANDOM_STATE)
else:
    CFG = {}
    MODELS = MODELS_DEFAULT
    MAX_ROWS_AFTER_FILTER = None



def find_tfidf_pair(cache_dir: str, n_train_rows: int, n_test_rows: int):

    train_files = [f for f in os.listdir(cache_dir) if f.startswith("tfidf_") and f.endswith("_X_train.npz")]
    candidates = []
    for tr in train_files:
        prefix = tr[:-len("_X_train.npz")]
        te = prefix + "_X_test.npz"
        tr_path = os.path.join(cache_dir, tr)
        te_path = os.path.join(cache_dir, te)
        if not os.path.exists(te_path):
            continue

        try:
            Xtr = load_npz(tr_path)
            Xte = load_npz(te_path)
            if Xtr.shape[0] == n_train_rows and Xte.shape[0] == n_test_rows:
                mtime = os.path.getmtime(tr_path)
                candidates.append((mtime, tr_path, te_path, Xtr.shape[1]))
        except Exception:
            continue

    if not candidates:
        raise RuntimeError("Could not find TF-IDF cache pair matching your train/test row counts.")

    candidates.sort(reverse=True, key=lambda x: x[0])  # newest first
    mtime, tr_path, te_path, n_feat = candidates[0]
    Xtr = load_npz(tr_path)
    Xte = load_npz(te_path)
    print(f"[cache] Using TF-IDF cache:\n  {os.path.basename(tr_path)}\n  {os.path.basename(te_path)}")
    print(f"[cache] TF-IDF features: {n_feat}")
    return Xtr, Xte


tfidf_train, tfidf_test = find_tfidf_pair(CACHE_DIR, len(idx_train), len(idx_test))


X_train = hstack([tfidf_train, csr_matrix(stylo_all[idx_train])], format="csr")
X_test  = hstack([tfidf_test,  csr_matrix(stylo_all[idx_test])],  format="csr")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)



domain_test = None

def try_load_domain_test():
    global domain_test
    # if already saved we use it
    dom_path = os.path.join(RUN_DIR, "domain_test.npy")
    if os.path.exists(dom_path):
        domain_test = np.load(dom_path, allow_pickle=True)
        print("[domain] Loaded domain_test.npy")
        return

    if not DO_DOMAIN_ERROR_ANALYSIS:
        return

    # attempt to read domain from TRAIN_PATH
    try:
        df = pd.read_csv(TRAIN_PATH, usecols=["model", "domain"])
    except Exception as e:
        print(f"[domain] Skipping domain error analysis (could not read model/domain): {e}")
        return

    df = df[df["model"].isin(MODELS)].reset_index(drop=True)

    # replicate optional cap if it was used
    if MAX_ROWS_AFTER_FILTER is not None and len(df) > MAX_ROWS_AFTER_FILTER:
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(
            df,
            train_size=MAX_ROWS_AFTER_FILTER,
            stratify=df["model"],
            random_state=RANDOM_STATE
        )
        df = df.reset_index(drop=True)

    # sanity check length matches stylo_all first dimension
    if len(df) != stylo_all.shape[0]:
        print(f"[domain] Skipping domain analysis: df length {len(df)} != stylo_all rows {stylo_all.shape[0]}")
        return

    domain_all = df["domain"].fillna("UNKNOWN").astype(str).values
    domain_test = domain_all[idx_test]
    np.save(dom_path, domain_test.astype(object))
    print("[domain] Saved domain_test.npy for reuse")


try_load_domain_test()



def evaluate(name, probs, y_true=y_test):
    preds = probs.argmax(axis=1)
    acc = accuracy_score(y_true, preds)
    f1  = f1_score(y_true, preds, average="macro")
    cm  = confusion_matrix(y_true, preds)

    print(f"\n{name}")
    print(f"{name} Accuracy: {acc:.6f}")
    print(f"{name} Macro F1:  {f1:.6f}")
    print(f"{name} Confusion Matrix:\n{cm}")

    return preds, {"name": name, "acc": float(acc), "macro_f1": float(f1)}

def error_analysis(name, preds, y_true=y_test):
    if domain_test is None:
        return
    y_true_names = np.array(classes, dtype=object)[y_true]
    y_pred_names = np.array(classes, dtype=object)[preds]

    errs = (preds != y_true)
    n_err = int(errs.sum())
    print(f"\nError analysis ({name}):")
    print("Number of errors:", n_err)
    if n_err == 0:
        return

    df = pd.DataFrame({
        "true_model": y_true_names[errs],
        "pred_model": y_pred_names[errs],
        "domain": np.array(domain_test, dtype=object)[errs],
    })
    top = (
        df.groupby(["true_model", "pred_model", "domain"])
          .size()
          .sort_values(ascending=False)
          .head(10)
    )
    print("\nTop confusions:")
    print(top)

def soft_vote(prob_list, weights):
    w = np.array(weights, dtype=np.float32)
    w = w / w.sum()
    out = np.zeros_like(prob_list[0], dtype=np.float32)
    for p, wi in zip(prob_list, w):
        out += wi * p.astype(np.float32)
    return out


def make_xgb_v2(n_estimators, seed):
    cfg = FULL_XGB_V2.copy()
    cfg["n_estimators"] = n_estimators
    return XGBClassifier(
        **cfg,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=N_JOBS
    )

def make_lgbm_v3(n_estimators, seed):
    cfg = FULL_LGBM_V3.copy()
    cfg["n_estimators"] = n_estimators
    return lgb.LGBMClassifier(
        **cfg,
        objective="multiclass",
        num_class=n_classes,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=-1,
        deterministic=True,
        force_row_wise=True
    )

def make_lr():
    return LogisticRegression(max_iter=4000, solver="lbfgs")

def make_sgd(seed):
    return SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        penalty="l2",
        max_iter=2000,
        tol=1e-3,
        random_state=seed
    )


def make_meta_logreg():
    return LogisticRegression(max_iter=5000, solver="lbfgs")

def make_meta_sgd(seed):
    return SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        penalty="l2",
        max_iter=500,
        tol=1e-3,
        random_state=seed,
        average=True
    )

def make_meta_extratrees(seed):
    return ExtraTreesClassifier(
        n_estimators=400,
        random_state=seed,
        n_jobs=N_JOBS
    )

def make_meta_lgbm(seed):
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=-1,
        deterministic=True,
        force_row_wise=True
    )

def make_meta_xgb(seed):
    return XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=N_JOBS
    )


#for voting + meta_test

print("\n[base] Training full base models (for voting + meta_test)...")
base_probs_test = {}

# LGBM_v3
m = make_lgbm_v3(n_estimators=FULL_LGBM_V3["n_estimators"], seed=RANDOM_STATE + 21)
m.fit(X_train, y_train)
p = m.predict_proba(X_test).astype(np.float32)
base_probs_test["LGBM_v3"] = p
preds, _ = evaluate("LGBM_v3", p); error_analysis("LGBM_v3", preds)
del m; gc.collect()

# XGB_v2
m = make_xgb_v2(n_estimators=FULL_XGB_V2["n_estimators"], seed=RANDOM_STATE)
m.fit(X_train, y_train)
p = m.predict_proba(X_test).astype(np.float32)
base_probs_test["XGB_v2"] = p
preds, _ = evaluate("XGB_v2", p); error_analysis("XGB_v2", preds)
del m; gc.collect()

# Logistic Regression base 
m.fit(X_train, y_train)
p = m.predict_proba(X_test).astype(np.float32)
base_probs_test["LR"] = p
preds, _ = evaluate("LR_base", p); error_analysis("LR_base", preds)
del m; gc.collect()

# SGD base 
m = make_sgd(seed=RANDOM_STATE)
m.fit(X_train, y_train)
p = m.predict_proba(X_test).astype(np.float32)
base_probs_test["SGD"] = p
preds, _ = evaluate("SGD_base", p); error_analysis("SGD_base", preds)
del m; gc.collect()


#voting

print("\n[vote] Running voting configurations...")
vote_results = []
for vname, (keys, weights) in VOTE_WEIGHTS.items():
    probs = soft_vote([base_probs_test[k] for k in keys], weights)
    preds, metrics = evaluate(vname, probs)
    error_analysis(vname, preds)
    vote_results.append(metrics)

#stacking
def get_base_model_for_oof(name, fold_seed):
    if name == "LGBM_v3":
        return make_lgbm_v3(n_estimators=STACK_LGBM_ESTIMATORS, seed=fold_seed)
    if name == "XGB_v2":
        return make_xgb_v2(n_estimators=STACK_XGB_ESTIMATORS, seed=fold_seed)
    if name == "LR":
        return make_lr()
    if name == "SGD":
        return make_sgd(seed=fold_seed)
    raise ValueError(f"Unknown base model: {name}")

def stack_run(stack_name, base_names, meta_name, cv_splits):
    print(f"\n[stack] {stack_name}")
    print(f"  Bases: {base_names}")
    print(f"  Meta:  {meta_name}  | CV={cv_splits}")

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros((X_train.shape[0], len(base_names) * n_classes), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
        print(f"  fold {fold}/{cv_splits}")
        Xtr, Xva = X_train[tr_idx], X_train[va_idx]
        ytr = y_train[tr_idx]

        fold_seed = RANDOM_STATE + 1000 + fold

        Ps = []
        for b in base_names:
            bm = get_base_model_for_oof(b, fold_seed + hash(b) % 10000)
            bm.fit(Xtr, ytr)
            Ps.append(bm.predict_proba(Xva).astype(np.float32))
            del bm
            gc.collect()

        oof[va_idx] = np.hstack(Ps).astype(np.float32)
        del Xtr, Xva, ytr, Ps
        gc.collect()

    meta_test = np.hstack([base_probs_test[b] for b in base_names]).astype(np.float32)

    # build meta model
    if meta_name == "LogReg":
        meta = make_meta_logreg()
    elif meta_name == "SGD_logloss":
        meta = make_meta_sgd(seed=RANDOM_STATE)
    elif meta_name == "ExtraTrees":
        meta = make_meta_extratrees(seed=RANDOM_STATE)
    elif meta_name == "LGBM":
        meta = make_meta_lgbm(seed=RANDOM_STATE)
    elif meta_name == "XGB":
        meta = make_meta_xgb(seed=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown meta: {meta_name}")

    print("  [meta] training...")
    meta.fit(oof, y_train)
    probs = meta.predict_proba(meta_test).astype(np.float32)

    preds, metrics = evaluate(stack_name, probs)
    error_analysis(stack_name, preds)

    del meta, oof, meta_test
    gc.collect()
    return metrics


stack_results = []

# CV=3 
stack_results.append(stack_run(
    "STACK_CV3__BASE_LGBM3_XGB2__META_LogReg",
    ["LGBM_v3", "XGB_v2"],
    "LogReg",
    CV3
))

stack_results.append(stack_run(
    "STACK_CV3__BASE_LGBM3_XGB2__META_SGDlogloss",
    ["LGBM_v3", "XGB_v2"],
    "SGD_logloss",
    CV3
))

stack_results.append(stack_run(
    "STACK_CV3__BASE_LGBM3_XGB2__META_ExtraTrees",
    ["LGBM_v3", "XGB_v2"],
    "ExtraTrees",
    CV3
))

stack_results.append(stack_run(
    "STACK_CV3__BASE_LGBM3_XGB2__META_XGB",
    ["LGBM_v3", "XGB_v2"],
    "XGB",
    CV3
))

stack_results.append(stack_run(
    "STACK_CV3__BASE_LGBM3_XGB2__META_LGBM",
    ["LGBM_v3", "XGB_v2"],
    "LGBM",
    CV3
))

stack_results.append(stack_run(
    "STACK_CV3__BASE_LGBM3_XGB2_LR_SGD__META_LogReg",
    ["LGBM_v3", "XGB_v2", "LR", "SGD"],
    "LogReg",
    CV3
))

# CV=5
stack_results.append(stack_run(
    "STACK_CV5__BASE_LGBM3_XGB2_LR_SGD__META_XGB",
    ["LGBM_v3", "XGB_v2", "LR", "SGD"],
    "XGB",
    CV5
))

stack_results.append(stack_run(
    "STACK_CV5__BASE_LGBM3_XGB2_LR_SGD__META_ExtraTrees",
    ["LGBM_v3", "XGB_v2", "LR", "SGD"],
    "ExtraTrees",
    CV5
))


print("\n=========================")
print("SUMMARY (Voting)")
print("=========================")
vote_results_sorted = sorted(vote_results, key=lambda d: d["macro_f1"], reverse=True)
for r in vote_results_sorted:
    print(f'{r["name"]}: acc={r["acc"]:.6f}  macroF1={r["macro_f1"]:.6f}')

print("\n=========================")
print("SUMMARY (Stacking)")
print("=========================")
stack_results_sorted = sorted(stack_results, key=lambda d: d["macro_f1"], reverse=True)
for r in stack_results_sorted:
    print(f'{r["name"]}: acc={r["acc"]:.6f}  macroF1={r["macro_f1"]:.6f}')

print("\nDone.")
