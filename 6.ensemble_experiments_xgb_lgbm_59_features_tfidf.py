import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import gc
import re
import json
import hashlib
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack, save_npz, load_npz
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier

from xgboost import XGBClassifier
import lightgbm as lgb

from joblib import dump, load

warnings.filterwarnings("ignore", message="X does not have valid feature names")

TRAIN_PATH = "raid_data/train_none.csv"


MODELS = ["cohere-chat", "gpt4", "mistral-chat", "mpt-chat", "llama-chat"]


RANDOM_STATE = 5
np.random.seed(RANDOM_STATE)


MAX_ROWS_AFTER_FILTER = None  

#TF-IDF config 
TFIDF_MAX_FEATURES = 2000
TFIDF_NGRAM_RANGE = (3, 5)
TFIDF_ANALYZER = "char"

#TF-IDF caching
TFIDF_USE_CACHE = True

#CPU threads
N_JOBS = 4


DO_VOTING = True
DO_OOF_STACKING = True

#Stacking folds for better OOF estimate
N_SPLITS = 3

STACK_XGB_ESTIMATORS = 180
STACK_LGBM_ESTIMATORS = 180

# Final full models 
FULL_XGB_ESTIMATORS = 300
FULL_LGBM_V2_ESTIMATORS = 450
FULL_LGBM_V3_ESTIMATORS = 650
FULL_LGBM_V4_ESTIMATORS = 500

# Output
RUN_DIR = os.path.join("runs", datetime.now().strftime("run_%Y%m%d_%H%M%S"))
os.makedirs(RUN_DIR, exist_ok=True)

CACHE_DIR = os.path.join("runs", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

STACK_BUNDLE_DIR = os.path.join(RUN_DIR, "stack_bundles")
os.makedirs(STACK_BUNDLE_DIR, exist_ok=True)


FEATURE_VERSION = "stylo59_base_names_v2"


FUNCTION_WORDS = {
    "the","and","to","of","in","that","is","it","for","on","with",
    "as","was","at","by","an","be","this","from","or","are"
}

FUNCTION_WORD_LIST = sorted(FUNCTION_WORDS)

def extract_repetition_features(text):
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return {"max_repeat_run": 0.0, "repeat_token_ratio": 0.0}

    max_run = 1
    run = 1
    repeats = 0

    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            run += 1
            max_run = max(max_run, run)
            repeats += 1
        else:
            run = 1

    return {
        "max_repeat_run": float(max_run),
        "repeat_token_ratio": float(repeats / len(words)),
    }

def extract_agreement_score(text):
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(sentences) < 2:
        return {"agreement_score": 0.0}

    token_sets = [
        set(re.findall(r"\b\w+\b", s.lower()))
        for s in sentences
    ]

    sims = []
    for i in range(len(token_sets) - 1):
        a, b = token_sets[i], token_sets[i + 1]
        if not a or not b:
            sims.append(0.0)
        else:
            sims.append(len(a & b) / len(a | b))

    return {"agreement_score": float(np.mean(sims))}

def extract_lexical_features(text):
    words = re.findall(r"\b\w+\b", text.lower())
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    num_chars = len(text)
    num_words = len(words)
    num_sentences = max(len(sentences), 1)

    word_lengths = [len(w) for w in words] if words else [0]
    vocab = set(words)
    vocab_size = len(vocab)

    counts = Counter(words)
    punct = re.findall(r"[.,;:!?]", text)

    if sentences:
        sent_lens = [len(s.split()) for s in sentences]
        max_sent_len = max(sent_lens)
        min_sent_len = min(sent_lens)
    else:
        max_sent_len = 0
        min_sent_len = 0

    return {
        "num_chars": num_chars,
        "num_words": num_words,
        "num_sentences": num_sentences,

        "avg_word_len": float(np.mean(word_lengths)),
        "std_word_len": float(np.std(word_lengths)),
        "min_word_len": float(np.min(word_lengths)),
        "max_word_len": float(np.max(word_lengths)),

        "avg_sentence_len": float(num_words / num_sentences),
        "max_sentence_len": float(max_sent_len),
        "min_sentence_len": float(min_sent_len),

        "vocab_size": float(vocab_size),
        "type_token_ratio": float(vocab_size / num_words) if num_words else 0.0,
        "hapax_ratio": float(sum(1 for w, c in counts.items() if c == 1) / num_words) if num_words else 0.0,

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
    words = re.findall(r"\b\w+\b", text.lower())
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
    words = re.findall(r"\b\w+\b", text.lower())
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
    feats.update(extract_agreement_score(text))
    feats.update(extract_repetition_features(text))
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
EXTRA_NAMES = ["agreement_score","max_repeat_run","repeat_token_ratio"]

FEATURE_NAMES = LEXICAL_NAMES + FW_NAMES + STRUCT_NAMES + STAT_NAMES + EXTRA_NAMES
N_STYLO = len(FEATURE_NAMES)
assert N_STYLO == 59, f"Expected 59 features, got {N_STYLO}"


#checks + caching

def check_probs(name, probs, n_rows, n_classes):
    if probs.shape != (n_rows, n_classes):
        raise ValueError(f"{name}: probs shape {probs.shape} != {(n_rows, n_classes)}")
    if not np.isfinite(probs).all():
        raise ValueError(f"{name}: probs has NaN/inf")
    s = probs.sum(axis=1)
    if not np.allclose(s, 1.0, atol=1e-3):
        raise ValueError(f"{name}: probs rows not summing to 1 (min={s.min()}, max={s.max()})")

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

def _md5_short(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()[:12]

def load_or_build_tfidf(texts_all, idx_train, idx_test):
    tfidf_params = {
        "analyzer": TFIDF_ANALYZER,
        "ngram_range": TFIDF_NGRAM_RANGE,
        "min_df": 3,
        "max_df": 0.95,
        "sublinear_tf": True,
        "max_features": TFIDF_MAX_FEATURES,
    }

    try:
        mtime = os.path.getmtime(TRAIN_PATH)
    except OSError:
        mtime = None

    key_payload = {
        "tfidf_params": tfidf_params,
        "MODELS": MODELS,
        "RANDOM_STATE": RANDOM_STATE,
        "MAX_ROWS_AFTER_FILTER": MAX_ROWS_AFTER_FILTER,
        "n_texts": int(len(texts_all)),
        "train_path": TRAIN_PATH,
        "mtime": mtime,
        "idx_train_md5": _md5_short(idx_train.tobytes()),
        "idx_test_md5": _md5_short(idx_test.tobytes()),
    }
    key_str = json.dumps(key_payload, sort_keys=True).encode("utf-8")
    cache_key = _md5_short(key_str)

    vec_path = os.path.join(CACHE_DIR, f"tfidf_{cache_key}_vectorizer.joblib")
    tr_path = os.path.join(CACHE_DIR, f"tfidf_{cache_key}_X_train.npz")
    te_path = os.path.join(CACHE_DIR, f"tfidf_{cache_key}_X_test.npz")

    if TFIDF_USE_CACHE and os.path.exists(vec_path) and os.path.exists(tr_path) and os.path.exists(te_path):
        try:
            tfidf = load(vec_path)
            X_tr = load_npz(tr_path).astype(np.float32)
            X_te = load_npz(te_path).astype(np.float32)
            # quick sanity
            if X_tr.shape[0] == len(idx_train) and X_te.shape[0] == len(idx_test):
                print(f"[cache] Loaded TF-IDF: key={cache_key}")
                return tfidf, X_tr, X_te
            else:
                print("[cache] TF-IDF cache shape mismatch -> rebuilding")
        except Exception as e:
            print(f"[cache] TF-IDF load failed ({e}) -> rebuilding")

    print("[build] TF-IDF (fit on train only)...")
    tfidf = TfidfVectorizer(**tfidf_params)
    X_tr = tfidf.fit_transform(texts_all[idx_train]).astype(np.float32)
    X_te = tfidf.transform(texts_all[idx_test]).astype(np.float32)

    if TFIDF_USE_CACHE:
        dump(tfidf, vec_path)
        save_npz(tr_path, X_tr)
        save_npz(te_path, X_te)
        print(f"[cache] Saved TF-IDF: key={cache_key}")

    return tfidf, X_tr, X_te

def eval_and_save(name: str, y_true: np.ndarray, probs: np.ndarray, out_dir: str):
    preds = np.argmax(probs, axis=1)
    acc = float(accuracy_score(y_true, preds))
    f1 = float(f1_score(y_true, preds, average="macro"))
    cm = confusion_matrix(y_true, preds)

    print(f"\n[{name}] Accuracy: {acc:.6f}")
    print(f"[{name}] Macro F1:  {f1:.6f}")
    print(f"[{name}] Confusion Matrix:\n{cm}")

    np.save(os.path.join(out_dir, f"{name}_preds.npy"), preds)
    np.save(os.path.join(out_dir, f"{name}_probs.npy"), probs.astype(np.float32))
    return {"name": name, "acc": acc, "macro_f1": f1}

def soft_vote(probs_list, weights=None):
    if weights is None:
        weights = [1.0] * len(probs_list)
    w = np.array(weights, dtype=np.float32)
    w = w / w.sum()
    out = np.zeros_like(probs_list[0], dtype=np.float32)
    for p, wi in zip(probs_list, w):
        out += wi * p.astype(np.float32)
    return out

def save_stack_bundle(stack_name: str, base_names, oof_train: np.ndarray, meta_test: np.ndarray,
                     y_train: np.ndarray, y_test: np.ndarray, out_dir: str):
    path = os.path.join(out_dir, f"{stack_name}.npz")
    np.savez_compressed(
        path,
        base_names=np.array(base_names, dtype=str),  
        oof_train=oof_train.astype(np.float32),
        meta_test=meta_test.astype(np.float32),
        y_train=y_train.astype(np.int32),
        y_test=y_test.astype(np.int32),
    )
    print(f"[stack] Saved bundle: {path}")



def make_xgb(n_estimators, n_classes, seed):
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=N_JOBS
    )

def make_lgbm_v2(n_classes, n_estimators, seed):
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        n_estimators=n_estimators,
        learning_rate=0.07,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=-1
    )

def make_lgbm_v3(n_classes, n_estimators, seed):
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=-1
    )

def make_lgbm_v4(n_classes, n_estimators, seed):
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        n_estimators=n_estimators,
        learning_rate=0.06,
        num_leaves=95,
        min_child_samples=50,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=-1
    )

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

    # Save split + labels mapping
    np.save(os.path.join(RUN_DIR, "idx_train.npy"), idx_train)
    np.save(os.path.join(RUN_DIR, "idx_test.npy"), idx_test)
    np.save(os.path.join(RUN_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(RUN_DIR, "y_test.npy"), y_test)
    with open(os.path.join(RUN_DIR, "classes.json"), "w") as f:
        json.dump(list(classes), f, indent=2)

    # Numeric feature cache key
    cache_key = f"{FEATURE_VERSION}_n{len(df)}_cap{MAX_ROWS_AFTER_FILTER or 'all'}_models{'-'.join(MODELS)}"
    numeric_cache = os.path.join(CACHE_DIR, f"stylo_{cache_key}.npy")

    X_num_all = build_numeric_for_all(texts_all, numeric_cache)
    np.save(os.path.join(RUN_DIR, "stylo_all.npy"), X_num_all)
    with open(os.path.join(RUN_DIR, "stylo_feature_names.json"), "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    Xn_train = X_num_all[idx_train]
    Xn_test = X_num_all[idx_test]

    #TF-IDF cached
    tfidf, X_tfidf_train, X_tfidf_test = load_or_build_tfidf(texts_all, idx_train, idx_test)
    print("TF-IDF train:", X_tfidf_train.shape, " test:", X_tfidf_test.shape)

    #combined
    X_train = hstack([X_tfidf_train, csr_matrix(Xn_train)], format="csr")
    X_test = hstack([X_tfidf_test, csr_matrix(Xn_test)], format="csr")
    print("Combined train:", X_train.shape, " test:", X_test.shape)

    del df, labels_all
    gc.collect()


    results = []
    probs_test = {}

    print("\n[train] XGB_combined...")
    xgb_full = make_xgb(FULL_XGB_ESTIMATORS, n_classes, seed=RANDOM_STATE)
    xgb_full.fit(X_train, y_train)
    p_xgb = xgb_full.predict_proba(X_test)
    check_probs("XGB_combined", p_xgb, len(y_test), n_classes)
    results.append(eval_and_save("XGB_combined", y_test, p_xgb, RUN_DIR))
    probs_test["XGB_combined"] = p_xgb

    print("\n[train] LGBM_combined_v2...")
    lgb_v2_full = make_lgbm_v2(n_classes, FULL_LGBM_V2_ESTIMATORS, seed=RANDOM_STATE + 13)
    lgb_v2_full.fit(X_train, y_train)
    p_v2 = lgb_v2_full.predict_proba(X_test)
    check_probs("LGBM_combined_v2", p_v2, len(y_test), n_classes)
    results.append(eval_and_save("LGBM_combined_v2", y_test, p_v2, RUN_DIR))
    probs_test["LGBM_combined_v2"] = p_v2

    print("\n[train] LGBM_combined_v3...")
    lgb_v3_full = make_lgbm_v3(n_classes, FULL_LGBM_V3_ESTIMATORS, seed=RANDOM_STATE + 21)
    lgb_v3_full.fit(X_train, y_train)
    p_v3 = lgb_v3_full.predict_proba(X_test)
    check_probs("LGBM_combined_v3", p_v3, len(y_test), n_classes)
    results.append(eval_and_save("LGBM_combined_v3", y_test, p_v3, RUN_DIR))
    probs_test["LGBM_combined_v3"] = p_v3

    print("\n[train] LGBM_combined_v4...")
    lgb_v4_full = make_lgbm_v4(n_classes, FULL_LGBM_V4_ESTIMATORS, seed=RANDOM_STATE + 33)
    lgb_v4_full.fit(X_train, y_train)
    p_v4 = lgb_v4_full.predict_proba(X_test)
    check_probs("LGBM_combined_v4", p_v4, len(y_test), n_classes)
    results.append(eval_and_save("LGBM_combined_v4", y_test, p_v4, RUN_DIR))
    probs_test["LGBM_combined_v4"] = p_v4

    with open(os.path.join(RUN_DIR, "base_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    if DO_VOTING:
        print("\n[vote] voting...")

        vote_sets = {
            "VOTE_v2_XGB": ["LGBM_combined_v2", "XGB_combined"],
            "VOTE_v2_v3_v4": ["LGBM_combined_v2", "LGBM_combined_v3", "LGBM_combined_v4"],
            "VOTE_v2_v3_v4_XGB": ["LGBM_combined_v2", "LGBM_combined_v3", "LGBM_combined_v4", "XGB_combined"],
            "VOTE_v2_v3": ["LGBM_combined_v2", "LGBM_combined_v3"],
            "VOTE_v2_v4": ["LGBM_combined_v2", "LGBM_combined_v4"],
        }

        vote_results = []
        for vname, mnames in vote_sets.items():
            probs = soft_vote([probs_test[m] for m in mnames])
            check_probs(vname, probs, len(y_test), n_classes)
            vote_results.append(eval_and_save(vname, y_test, probs, RUN_DIR))

        with open(os.path.join(RUN_DIR, "voting_results.json"), "w") as f:
            json.dump(vote_results, f, indent=2)


    if DO_OOF_STACKING:
        print("\n[stack] stacking (base = v2/v3/v4/XGB) ...")

        base_names = ["LGBM_combined_v2", "LGBM_combined_v3", "LGBM_combined_v4", "XGB_combined"]
        oof = np.zeros((X_train.shape[0], len(base_names) * n_classes), dtype=np.float32)

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
            print(f"  fold {fold}/{N_SPLITS}")

            Xtr, Xva = X_train[tr_idx], X_train[val_idx]
            ytr = y_train[tr_idx]

            m_v2 = make_lgbm_v2(n_classes, STACK_LGBM_ESTIMATORS, seed=RANDOM_STATE + 100 + fold)
            m_v3 = make_lgbm_v3(n_classes, STACK_LGBM_ESTIMATORS, seed=RANDOM_STATE + 200 + fold)
            m_v4 = make_lgbm_v4(n_classes, STACK_LGBM_ESTIMATORS, seed=RANDOM_STATE + 300 + fold)
            m_xg = make_xgb(STACK_XGB_ESTIMATORS, n_classes, seed=RANDOM_STATE + 400 + fold)

            m_v2.fit(Xtr, ytr)
            m_v3.fit(Xtr, ytr)
            m_v4.fit(Xtr, ytr)
            m_xg.fit(Xtr, ytr)

            P = np.hstack([
                m_v2.predict_proba(Xva),
                m_v3.predict_proba(Xva),
                m_v4.predict_proba(Xva),
                m_xg.predict_proba(Xva),
            ]).astype(np.float32)

            if P.shape != (len(val_idx), len(base_names) * n_classes):
                raise ValueError(f"OOF fold {fold}: bad P shape {P.shape}")

            oof[val_idx] = P

            del m_v2, m_v3, m_v4, m_xg, Xtr, Xva, ytr, P
            gc.collect()

        meta_test = np.hstack([
            probs_test["LGBM_combined_v2"],
            probs_test["LGBM_combined_v3"],
            probs_test["LGBM_combined_v4"],
            probs_test["XGB_combined"],
        ]).astype(np.float32)

        save_stack_bundle(
            "STACK_BASE_v2v3v4XGB",
            base_names,
            oof,
            meta_test,
            y_train,
            y_test,
            STACK_BUNDLE_DIR
        )

        meta_models = [
            ("META_LogReg", LogisticRegression(max_iter=4000, solver="lbfgs")),
            ("META_SGD_logloss", SGDClassifier(
                loss="log_loss", alpha=1e-4, penalty="l2",
                max_iter=150, tol=1e-3,
                random_state=RANDOM_STATE, average=True
            )),
            ("META_ExtraTrees", ExtraTreesClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=N_JOBS)),
            ("META_LGBM", lgb.LGBMClassifier(
                objective="multiclass", num_class=n_classes,
                n_estimators=600, learning_rate=0.05, num_leaves=31,
                random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=-1
            )),
            ("META_XGB", XGBClassifier(
                n_estimators=600, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                objective="multi:softprob", num_class=n_classes,
                eval_metric="mlogloss", tree_method="hist",
                random_state=RANDOM_STATE, n_jobs=N_JOBS
            )),
        ]

        stack_results = []
        for name, meta in meta_models:
            print(f"\n[stack-meta] Training {name}...")
            meta.fit(oof, y_train)
            sp = meta.predict_proba(meta_test)
            check_probs(f"STACK_v2v3v4XGB__{name}", sp, len(y_test), n_classes)
            stack_results.append(eval_and_save(f"STACK_v2v3v4XGB__{name}", y_test, sp, RUN_DIR))
            del meta
            gc.collect()

        with open(os.path.join(RUN_DIR, "stacking_results.json"), "w") as f:
            json.dump(stack_results, f, indent=2)


    with open(os.path.join(RUN_DIR, "config.json"), "w") as f:
        json.dump({
            "TRAIN_PATH": TRAIN_PATH,
            "MODELS": MODELS,
            "RANDOM_STATE": RANDOM_STATE,
            "MAX_ROWS_AFTER_FILTER": MAX_ROWS_AFTER_FILTER,
            "TFIDF_MAX_FEATURES": TFIDF_MAX_FEATURES,
            "TFIDF_NGRAM_RANGE": TFIDF_NGRAM_RANGE,
            "TFIDF_ANALYZER": TFIDF_ANALYZER,
            "TFIDF_USE_CACHE": TFIDF_USE_CACHE,
            "N_JOBS": N_JOBS,
            "N_SPLITS": N_SPLITS,
            "STACK_XGB_ESTIMATORS": STACK_XGB_ESTIMATORS,
            "STACK_LGBM_ESTIMATORS": STACK_LGBM_ESTIMATORS,
            "FEATURE_VERSION": FEATURE_VERSION,
            "N_STYLO": N_STYLO,
        }, f, indent=2)

    print("\nDone. Everything saved in:", RUN_DIR)

if __name__ == "__main__":
    main()
