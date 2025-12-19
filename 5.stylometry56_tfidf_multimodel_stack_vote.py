import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import gc
import re
import json
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB

from xgboost import XGBClassifier
import lightgbm as lgb

warnings.filterwarnings("ignore", message="X does not have valid feature names")



TRAIN_PATH = "raid_data/train_none.csv"
MODELS = ["cohere-chat", "gpt4", "mistral-chat", "mpt-chat", "llama-chat"]

RANDOM_STATE = 5
np.random.seed(RANDOM_STATE)

MAX_ROWS_AFTER_FILTER = None  

TFIDF_MAX_FEATURES = 2000
TFIDF_NGRAM_RANGE = (3, 5)
TFIDF_ANALYZER = "char"

N_JOBS = 4

DO_VOTING = True
DO_OOF_STACKING = True
N_SPLITS = 3

STACK_XGB_ESTIMATORS = 180
STACK_LGBM_ESTIMATORS = 180

# Output
RUN_DIR = os.path.join("runs", datetime.now().strftime("run_%Y%m%d_%H%M%S"))
os.makedirs(RUN_DIR, exist_ok=True)

CACHE_DIR = os.path.join("runs", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)



FUNCTION_WORDS = {
    "the","and","to","of","in","that","is","it","for","on","with",
    "as","was","at","by","an","be","this","from","or","are"
}
FUNCTION_WORD_LIST = sorted(FUNCTION_WORDS)

_word_re = re.compile(r"\b\w+\b")
_sent_re = re.compile(r"[.!?]+")
_punct_re = re.compile(r"[.,;:!?]")


def extract_lexical_features(text):
    words = _word_re.findall(text.lower())
    sentences = [s for s in _sent_re.split(text) if s.strip()]
    num_chars = len(text)
    num_words = len(words)
    num_sentences = max(len(sentences), 1)

    word_lengths = [len(w) for w in words] if words else [0]
    vocab = set(words)
    vocab_size = len(vocab)

    counts = Counter(words)
    punct = _punct_re.findall(text)

    if sentences:
        sent_lens = [len(s.split()) for s in sentences]
        max_sent_len = max(sent_lens)
        min_sent_len = min(sent_lens)
    else:
        max_sent_len = 0
        min_sent_len = 0

    return {
        "num_chars": float(num_chars),
        "num_words": float(num_words),
        "num_sentences": float(num_sentences),

        "avg_word_len": float(np.mean(word_lengths)),
        "std_word_len": float(np.std(word_lengths)),
        "min_word_len": float(np.min(word_lengths)),
        "max_word_len": float(np.max(word_lengths)),

        "avg_sentence_len": float(num_words / num_sentences),
        "max_sentence_len": float(max_sent_len),
        "min_sentence_len": float(min_sent_len),

        "vocab_size": float(vocab_size),
        "type_token_ratio": float(vocab_size / num_words) if num_words else 0.0,
        "hapax_ratio": float(sum(1 for _, c in counts.items() if c == 1) / num_words) if num_words else 0.0,

        "uppercase_ratio": float(sum(1 for c in text if c.isupper()) / max(num_chars, 1)),

        "punct_ratio": float(len(punct) / max(num_chars, 1)),
        "comma_ratio": float(text.count(",") / max(num_chars, 1)),
        "period_ratio": float(text.count(".") / max(num_chars, 1)),
        "exclamation_ratio": float(text.count("!") / max(num_chars, 1)),
        "question_ratio": float(text.count("?") / max(num_chars, 1)),

        "digit_ratio": float(sum(1 for c in text if c.isdigit()) / max(num_chars, 1)),
        "whitespace_ratio": float(text.count(" ") / max(num_chars, 1)),
    }


def extract_function_word_features(text):
    words = _word_re.findall(text.lower())
    num_words = len(words) if words else 1
    counts = Counter(words)

    feats = {}
    for fw in FUNCTION_WORD_LIST:
        feats[f"fw_{fw}_ratio"] = float(counts.get(fw, 0) / num_words)
    return feats


def extract_structure_features(text):
    paragraphs = [p for p in text.split("\n") if p.strip()]
    num_chars = max(len(text), 1)

    return {
        "num_paragraphs": float(len(paragraphs)),
        "avg_paragraph_len": float(np.mean([len(p.split()) for p in paragraphs])) if paragraphs else 0.0,
        "newline_ratio": float(text.count("\n") / num_chars),
        "quote_ratio": float(text.count('"') / num_chars),
        "parentheses_ratio": float((text.count("(") + text.count(")")) / num_chars),
        "dash_ratio": float(text.count("-") / num_chars),
        "semicolon_ratio": float(text.count(";") / num_chars),
        "colon_ratio": float(text.count(":") / num_chars),
    }


def extract_statistical_features(text):
    words = _word_re.findall(text.lower())
    num_words = len(words) if words else 1
    counts = Counter(words)

    if len(counts) == 0:
        return {
            "token_entropy": 0.0,
            "unique_ratio": 0.0,
            "repeated_type_ratio": 0.0,
            "avg_token_freq": 0.0,
            "max_token_freq": 0.0,
            "std_token_freq": 0.0,
        }

    freqs = np.array(list(counts.values()), dtype=np.float64) / num_words
    entropy = -np.sum(freqs * np.log2(freqs + 1e-12))
    repeated_types = sum(1 for c in counts.values() if c > 5)

    return {
        "token_entropy": float(entropy),
        "unique_ratio": float(len(counts) / num_words),
        "repeated_type_ratio": float(repeated_types / num_words),
        "avg_token_freq": float(np.mean(freqs)),
        "max_token_freq": float(np.max(freqs)),
        "std_token_freq": float(np.std(freqs)),
    }


def extract_all_features(text):
    feats = {}
    feats.update(extract_lexical_features(text))
    feats.update(extract_function_word_features(text))
    feats.update(extract_structure_features(text))
    feats.update(extract_statistical_features(text))
    return feats


LEXICAL_NAMES = [
    "num_chars","num_words","num_sentences",
    "avg_word_len","std_word_len","min_word_len","max_word_len",
    "avg_sentence_len","max_sentence_len","min_sentence_len",
    "vocab_size","type_token_ratio","hapax_ratio",
    "uppercase_ratio",
    "punct_ratio","comma_ratio","period_ratio","exclamation_ratio","question_ratio",
    "digit_ratio","whitespace_ratio",
]
FW_NAMES = [f"fw_{fw}_ratio" for fw in FUNCTION_WORD_LIST]
STRUCT_NAMES = [
    "num_paragraphs","avg_paragraph_len","newline_ratio","quote_ratio",
    "parentheses_ratio","dash_ratio","semicolon_ratio","colon_ratio",
]
STAT_NAMES = [
    "token_entropy","unique_ratio","repeated_type_ratio",
    "avg_token_freq","max_token_freq","std_token_freq",
]
FEATURE_NAMES = LEXICAL_NAMES + FW_NAMES + STRUCT_NAMES + STAT_NAMES
assert len(FEATURE_NAMES) == 56
N_STYLO = 56


def build_numeric_for_all(texts: np.ndarray, cache_path: str) -> np.ndarray:
    if os.path.exists(cache_path):
        X = np.load(cache_path)
        if X.shape == (len(texts), N_STYLO):
            print(f"[cache] Loaded numeric features: {cache_path}")
            return X

    print(f"[build] Computing {N_STYLO} numeric features (cached)...")
    X = np.zeros((len(texts), N_STYLO), dtype=np.float32)

    for i, t in enumerate(texts):
        if i % 5000 == 0:
            print(f"  Processed {i}/{len(texts)}")
        d = extract_all_features(t)
        for j, k in enumerate(FEATURE_NAMES):
            X[i, j] = float(d.get(k, 0.0))

    np.save(cache_path, X)
    return X


def make_lgbm(n_estimators: int, n_classes: int):
    kwargs = dict(
        objective="multiclass",
        num_class=n_classes,
        n_estimators=n_estimators,
        learning_rate=0.1,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbosity=-1,
    )
    try:
        return lgb.LGBMClassifier(**kwargs, force_col_wise=True)
    except TypeError:
        return lgb.LGBMClassifier(**kwargs)


def eval_and_save(name: str, y_true: np.ndarray, probs: np.ndarray, out_dir: str):
    preds = np.argmax(probs, axis=1)
    acc = float(accuracy_score(y_true, preds))
    f1 = float(f1_score(y_true, preds, average="macro"))
    cm = confusion_matrix(y_true, preds)

    print(f"\n[{name}] Accuracy: {acc:.6f}")
    print(f"[{name}] Macro F1:  {f1:.6f}")
    print(f"[{name}] Confusion Matrix:\n{cm}")

    np.save(os.path.join(out_dir, f"{name}_preds.npy"), preds)
    np.save(os.path.join(out_dir, f"{name}_probs.npy"), probs)
    return {"name": name, "acc": acc, "macro_f1": f1}


def complement_report(A_name, A_preds, B_name, B_preds, y_true):
    A = A_preds
    B = B_preds
    y = y_true
    disagree = float(np.mean(A != B))
    B_helps_A = float(np.mean((A != y) & (B == y)))
    A_helps_B = float(np.mean((B != y) & (A == y)))
    oracle = float(np.mean((A == y) | (B == y)))
    return {
        "pair": f"{A_name} vs {B_name}",
        "disagree_rate": disagree,
        "B_helps_A": B_helps_A,
        "A_helps_B": A_helps_B,
        "oracle_upper_bound": oracle,
    }


def soft_vote(probs_list, weights=None):
    if weights is None:
        weights = [1.0] * len(probs_list)
    w = np.array(weights, dtype=np.float32)
    w = w / w.sum()
    out = np.zeros_like(probs_list[0], dtype=np.float32)
    for p, wi in zip(probs_list, w):
        out += wi * p.astype(np.float32)
    return out


def weight_search(val_probs, val_y, model_names, grid):
    best = None
    for wt in grid:
        probs = soft_vote([val_probs[n] for n in model_names], weights=wt)
        preds = np.argmax(probs, axis=1)
        f1 = float(f1_score(val_y, preds, average="macro"))
        acc = float(accuracy_score(val_y, preds))
        if (best is None) or (f1 > best["macro_f1"]):
            best = {"weights": wt, "macro_f1": f1, "acc": acc}
    return best


def main():
    print(f"[run] Output folder: {RUN_DIR}")

    df = pd.read_csv(TRAIN_PATH, usecols=["generation", "model"])
    df = df[df["model"].isin(MODELS)].reset_index(drop=True)

    if MAX_ROWS_AFTER_FILTER is not None and len(df) > MAX_ROWS_AFTER_FILTER:
        df, _ = train_test_split(
            df,
            train_size=MAX_ROWS_AFTER_FILTER,
            stratify=df["model"],
            random_state=RANDOM_STATE
        )
        df = df.reset_index(drop=True)

    print("\nDATASET SIZE INFO")
    print("After filtering/cap:", len(df))
    print(df["model"].value_counts())

    texts_all = df["generation"].fillna("").astype(str).values
    labels_all = df["model"].astype(str).values

    le = LabelEncoder()
    y_all = le.fit_transform(labels_all)
    classes = le.classes_
    n_classes = len(classes)
    print("\nClasses:", list(classes))

    idx_all = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=0.2, stratify=y_all, random_state=RANDOM_STATE
    )
    y_train = y_all[idx_train]
    y_test = y_all[idx_test]


    cache_key = f"n{len(df)}_cap{MAX_ROWS_AFTER_FILTER or 'all'}_models{'-'.join(MODELS)}"
    numeric_cache = os.path.join(CACHE_DIR, f"numeric56_{cache_key}.npy")

    X_num_all = build_numeric_for_all(texts_all, numeric_cache)
    np.save(os.path.join(RUN_DIR, "numeric56_all.npy"), X_num_all)

    X56_train = X_num_all[idx_train]
    X56_test = X_num_all[idx_test]


    print("\n[build] TF-IDF...")
    tfidf = TfidfVectorizer(
        analyzer=TFIDF_ANALYZER,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        max_features=TFIDF_MAX_FEATURES
    )
    X_tfidf_train = tfidf.fit_transform(texts_all[idx_train]).astype(np.float32)
    X_tfidf_test = tfidf.transform(texts_all[idx_test]).astype(np.float32)
    print("TF-IDF train:", X_tfidf_train.shape, " test:", X_tfidf_test.shape)

    X_train = hstack([X_tfidf_train, csr_matrix(X56_train)], format="csr")
    X_test = hstack([X_tfidf_test, csr_matrix(X56_test)], format="csr")
    print("Combined train:", X_train.shape, " test:", X_test.shape)

    del df, labels_all
    gc.collect()

    results = []
    probs_test = {}
    preds_test = {}

    print("\n[train] LGBM_combined...")
    lgbm_comb = make_lgbm(n_estimators=300, n_classes=n_classes)
    lgbm_comb.fit(X_train, y_train)
    p_lgbm_comb = lgbm_comb.predict_proba(X_test)
    results.append(eval_and_save("LGBM_combined", y_test, p_lgbm_comb, RUN_DIR))
    probs_test["LGBM_combined"] = p_lgbm_comb
    preds_test["LGBM_combined"] = np.argmax(p_lgbm_comb, axis=1)

    print("\n[train] XGB_combined...")
    xgb_comb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )
    xgb_comb.fit(X_train, y_train)
    p_xgb_comb = xgb_comb.predict_proba(X_test)
    results.append(eval_and_save("XGB_combined", y_test, p_xgb_comb, RUN_DIR))
    probs_test["XGB_combined"] = p_xgb_comb
    preds_test["XGB_combined"] = np.argmax(p_xgb_comb, axis=1)

    print("\n[train] LGBM_combined_v2...")
    lgbm_comb_v2 = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        n_estimators=450,
        learning_rate=0.07,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE + 13,
        n_jobs=N_JOBS,
        verbosity=-1
    )
    lgbm_comb_v2.fit(X_train, y_train)
    p_lgbm_v2 = lgbm_comb_v2.predict_proba(X_test)
    results.append(eval_and_save("LGBM_combined_v2", y_test, p_lgbm_v2, RUN_DIR))
    probs_test["LGBM_combined_v2"] = p_lgbm_v2
    preds_test["LGBM_combined_v2"] = np.argmax(p_lgbm_v2, axis=1)

    print("\n[train] SGD_tfidf...")
    sgd = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        penalty="l2",
        max_iter=30,
        tol=1e-3,
        random_state=RANDOM_STATE,
        average=True
    )
    sgd.fit(X_tfidf_train, y_train)
    p_sgd = sgd.predict_proba(X_tfidf_test)
    results.append(eval_and_save("SGD_tfidf", y_test, p_sgd, RUN_DIR))
    probs_test["SGD_tfidf"] = p_sgd
    preds_test["SGD_tfidf"] = np.argmax(p_sgd, axis=1)

    print("\n[train] CNB_tfidf...")
    cnb = ComplementNB(alpha=0.5)
    cnb.fit(X_tfidf_train, y_train)
    p_cnb = cnb.predict_proba(X_tfidf_test)
    results.append(eval_and_save("CNB_tfidf", y_test, p_cnb, RUN_DIR))
    probs_test["CNB_tfidf"] = p_cnb
    preds_test["CNB_tfidf"] = np.argmax(p_cnb, axis=1)

    print("\n[train] LGBM_56only...")
    lgbm_56 = make_lgbm(n_estimators=300, n_classes=n_classes)
    lgbm_56.fit(X56_train, y_train)
    p_lgbm_56 = lgbm_56.predict_proba(X56_test)
    results.append(eval_and_save("LGBM_56only", y_test, p_lgbm_56, RUN_DIR))
    probs_test["LGBM_56only"] = p_lgbm_56
    preds_test["LGBM_56only"] = np.argmax(p_lgbm_56, axis=1)

    with open(os.path.join(RUN_DIR, "base_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n[analysis] Complementarity (pairwise):")
    comp = []
    names = list(preds_test.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = complement_report(names[i], preds_test[names[i]], names[j], preds_test[names[j]], y_test)
            comp.append(r)
            print(r)
    with open(os.path.join(RUN_DIR, "complementarity.json"), "w") as f:
        json.dump(comp, f, indent=2)

    if DO_VOTING:
        print("\n[vote] Equal votes...")
        vote_sets = {
            "VOTE_LGBM+XGB": ["LGBM_combined", "XGB_combined"],
            "VOTE_LGBM+XGB+SGD": ["LGBM_combined", "XGB_combined", "SGD_tfidf"],
            "VOTE_LGBM+XGB+SGD+CNB": ["LGBM_combined", "XGB_combined", "SGD_tfidf", "CNB_tfidf"],
            "VOTE_LGBM+LGBMv2+XGB": ["LGBM_combined", "LGBM_combined_v2", "XGB_combined"],
            "VOTE_LGBM+LGBMv2+XGB+SGD": ["LGBM_combined", "LGBM_combined_v2", "XGB_combined", "SGD_tfidf"],
        }

        vote_results = []
        for vname, mnames in vote_sets.items():
            probs = soft_vote([probs_test[m] for m in mnames])
            vote_results.append(eval_and_save(vname, y_test, probs, RUN_DIR))

        print("\n[vote] Weight search (train-only validation)...")
        tr2_idx, val2_idx = train_test_split(
            np.arange(X_train.shape[0]),
            test_size=0.12,
            stratify=y_train,
            random_state=RANDOM_STATE
        )

        X_tr2, y_tr2 = X_train[tr2_idx], y_train[tr2_idx]
        X_val2, y_val2 = X_train[val2_idx], y_train[val2_idx]
        Xtf_tr2, Xtf_val2 = X_tfidf_train[tr2_idx], X_tfidf_train[val2_idx]

        lgb_val = make_lgbm(n_estimators=220, n_classes=n_classes)
        xgb_val = XGBClassifier(
            n_estimators=220, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=n_classes,
            eval_metric="mlogloss", tree_method="hist",
            random_state=RANDOM_STATE, n_jobs=N_JOBS
        )
        sgd_val = SGDClassifier(
            loss="log_loss", alpha=1e-4, penalty="l2",
            max_iter=25, tol=1e-3,
            random_state=RANDOM_STATE, average=True
        )
        cnb_val = ComplementNB(alpha=0.5)

        lgb_val.fit(X_tr2, y_tr2)
        xgb_val.fit(X_tr2, y_tr2)
        sgd_val.fit(Xtf_tr2, y_tr2)
        cnb_val.fit(Xtf_tr2, y_tr2)

        val_probs = {
            "LGBM": lgb_val.predict_proba(X_val2),
            "XGB": xgb_val.predict_proba(X_val2),
            "SGD": sgd_val.predict_proba(Xtf_val2),
            "CNB": cnb_val.predict_proba(Xtf_val2),
        }

        grid2 = [(1,1),(2,1),(1,2),(3,1),(1,3)]
        grid3 = [
            (1,1,1),(2,1,1),(1,2,1),(1,1,2),
            (2,2,1),(2,1,2),(1,2,2),
            (3,1,1),(1,3,1),(1,1,3)
        ]
        grid4 = [
            (1,1,1,1),
            (2,1,1,1),(1,2,1,1),(1,1,2,1),(1,1,1,2),
            (2,2,1,1),(2,1,2,1),(2,1,1,2),
            (1,2,2,1),(1,2,1,2),(1,1,2,2)
        ]

        best_2 = weight_search(val_probs, y_val2, ["LGBM","XGB"], grid2)
        best_3 = weight_search(val_probs, y_val2, ["LGBM","XGB","SGD"], grid3)
        best_4 = weight_search(val_probs, y_val2, ["LGBM","XGB","SGD","CNB"], grid4)

        print("Best weights (LGBM,XGB):", best_2)
        print("Best weights (LGBM,XGB,SGD):", best_3)
        print("Best weights (LGBM,XGB,SGD,CNB):", best_4)

        tuned_2 = soft_vote([probs_test["LGBM_combined"], probs_test["XGB_combined"]], weights=best_2["weights"])
        vote_results.append(eval_and_save("VOTE_TUNED_LGBM+XGB", y_test, tuned_2, RUN_DIR))

        tuned_3 = soft_vote(
            [probs_test["LGBM_combined"], probs_test["XGB_combined"], probs_test["SGD_tfidf"]],
            weights=best_3["weights"]
        )
        vote_results.append(eval_and_save("VOTE_TUNED_LGBM+XGB+SGD", y_test, tuned_3, RUN_DIR))

        tuned_4 = soft_vote(
            [probs_test["LGBM_combined"], probs_test["XGB_combined"], probs_test["SGD_tfidf"], probs_test["CNB_tfidf"]],
            weights=best_4["weights"]
        )
        vote_results.append(eval_and_save("VOTE_TUNED_LGBM+XGB+SGD+CNB", y_test, tuned_4, RUN_DIR))

        with open(os.path.join(RUN_DIR, "voting_results.json"), "w") as f:
            json.dump(vote_results, f, indent=2)

        del lgb_val, xgb_val, sgd_val, cnb_val
        del X_tr2, X_val2, y_tr2, y_val2, Xtf_tr2, Xtf_val2
        gc.collect()

    if DO_OOF_STACKING:
        print("\n[stack] OOF stacking (meta = SGD log_loss)...")

        def oof_stack(base_list, stack_name):
            oof = np.zeros((X_train.shape[0], len(base_list) * n_classes), dtype=np.float32)
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
                print(f"  {stack_name} fold {fold}/{N_SPLITS}")

                Xtr_c, Xva_c = X_train[tr_idx], X_train[val_idx]
                ytr = y_train[tr_idx]

                Xtr_t, Xva_t = X_tfidf_train[tr_idx], X_tfidf_train[val_idx]
                Xtr_56, Xva_56 = X56_train[tr_idx], X56_train[val_idx]

                fold_probs = []
                for kind in base_list:
                    if kind == "combined_lgbm":
                        m = make_lgbm(n_estimators=STACK_LGBM_ESTIMATORS, n_classes=n_classes)
                        m.fit(Xtr_c, ytr)
                        fold_probs.append(m.predict_proba(Xva_c))
                    elif kind == "combined_xgb":
                        m = XGBClassifier(
                            n_estimators=STACK_XGB_ESTIMATORS,
                            max_depth=6,
                            learning_rate=0.1,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            objective="multi:softprob",
                            num_class=n_classes,
                            eval_metric="mlogloss",
                            tree_method="hist",
                            random_state=RANDOM_STATE,
                            n_jobs=N_JOBS
                        )
                        m.fit(Xtr_c, ytr)
                        fold_probs.append(m.predict_proba(Xva_c))
                    elif kind == "tfidf_sgd":
                        m = SGDClassifier(
                            loss="log_loss", alpha=1e-4, penalty="l2",
                            max_iter=25, tol=1e-3,
                            random_state=RANDOM_STATE, average=True
                        )
                        m.fit(Xtr_t, ytr)
                        fold_probs.append(m.predict_proba(Xva_t))
                    elif kind == "tfidf_cnb":
                        m = ComplementNB(alpha=0.5)
                        m.fit(Xtr_t, ytr)
                        fold_probs.append(m.predict_proba(Xva_t))
                    elif kind == "only56_lgbm":
                        m = make_lgbm(n_estimators=STACK_LGBM_ESTIMATORS, n_classes=n_classes)
                        m.fit(Xtr_56, ytr)
                        fold_probs.append(m.predict_proba(Xva_56))
                    else:
                        raise ValueError("Unknown kind: " + kind)

                    del m
                    gc.collect()

                oof[val_idx] = np.hstack(fold_probs).astype(np.float32)

                del Xtr_c, Xva_c, Xtr_t, Xva_t, Xtr_56, Xva_56, ytr
                gc.collect()

            meta = SGDClassifier(
                loss="log_loss",
                alpha=1e-4,
                penalty="l2",
                max_iter=80,
                tol=1e-3,
                random_state=RANDOM_STATE,
                average=True
            )
            meta.fit(oof, y_train)

            test_feats = []
            for kind in base_list:
                if kind == "combined_lgbm":
                    test_feats.append(probs_test["LGBM_combined"])
                elif kind == "combined_xgb":
                    test_feats.append(probs_test["XGB_combined"])
                elif kind == "tfidf_sgd":
                    test_feats.append(probs_test["SGD_tfidf"])
                elif kind == "tfidf_cnb":
                    test_feats.append(probs_test["CNB_tfidf"])
                elif kind == "only56_lgbm":
                    test_feats.append(probs_test["LGBM_56only"])
                else:
                    raise ValueError("Unknown kind: " + kind)

            meta_test = np.hstack(test_feats).astype(np.float32)
            return meta.predict_proba(meta_test)

        stack_results = []
        probs_s1 = oof_stack(["combined_lgbm", "combined_xgb"], "STACK_LGBM+XGB")
        stack_results.append(eval_and_save("STACK_LGBM+XGB", y_test, probs_s1, RUN_DIR))

        probs_s2 = oof_stack(["combined_lgbm", "combined_xgb", "tfidf_sgd", "tfidf_cnb"], "STACK_LGBM+XGB+SGD+CNB")
        stack_results.append(eval_and_save("STACK_LGBM+XGB+SGD+CNB", y_test, probs_s2, RUN_DIR))

        probs_s3 = oof_stack(["combined_lgbm", "only56_lgbm"], "STACK_LGBMcombined+LGBM56")
        stack_results.append(eval_and_save("STACK_LGBMcombined+LGBM56", y_test, probs_s3, RUN_DIR))

        with open(os.path.join(RUN_DIR, "stacking_results.json"), "w") as f:
            json.dump(stack_results, f, indent=2)

    np.save(os.path.join(RUN_DIR, "y_test.npy"), y_test)
    with open(os.path.join(RUN_DIR, "classes.json"), "w") as f:
        json.dump(list(classes), f, indent=2)


    with open(os.path.join(RUN_DIR, "config.json"), "w") as f:
        json.dump({
            "TRAIN_PATH": TRAIN_PATH,
            "MODELS": MODELS,
            "RANDOM_STATE": RANDOM_STATE,
            "MAX_ROWS_AFTER_FILTER": MAX_ROWS_AFTER_FILTER,
            "TFIDF_MAX_FEATURES": TFIDF_MAX_FEATURES,
            "TFIDF_NGRAM_RANGE": TFIDF_NGRAM_RANGE,
            "TFIDF_ANALYZER": TFIDF_ANALYZER,
            "N_JOBS": N_JOBS,
            "N_SPLITS": N_SPLITS,
            "STACK_XGB_ESTIMATORS": STACK_XGB_ESTIMATORS,
            "STACK_LGBM_ESTIMATORS": STACK_LGBM_ESTIMATORS,
        }, f, indent=2)

    print("\nDone. Everything saved in:", RUN_DIR)


if __name__ == "__main__":
    main()
