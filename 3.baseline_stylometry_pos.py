import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
import lightgbm as lgb

import spacy


nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


TRAIN_PATH = "raid_data/train_none.csv"

# not used now; we take all data for the top-K models
MAX_TRAIN = 20000

#top-K models by frequency (set to None to keep all)
TOP_K_MODELS = 10
MODELS =["cohere-chat", "gpt4", "mistral-chat", "mpt-chat", "llama-chat"]

RANDOM_STATE = 5
#SHAP analysis
RUN_SHAP = False

# FEATUREs
FUNCTION_WORDS = {
    "the","and","to","of","in","that","is","it","for","on","with",
    "as","was","at","by","an","be","this","from","or","are"
}

def extract_pos_features_doc(doc):
    pos_tags = [tok.pos_ for tok in doc if tok.is_alpha]

    if not pos_tags:
        return {
            "pos_noun_ratio": 0.0,
            "pos_verb_ratio": 0.0,
            "pos_adj_ratio": 0.0,
            "pos_adv_ratio": 0.0,
            "pos_pron_ratio": 0.0,
            "pos_func_ratio": 0.0,
            "pos_entropy": 0.0,
        }

    total = len(pos_tags)
    counts = Counter(pos_tags)

    noun = counts.get("NOUN", 0) + counts.get("PROPN", 0)
    verb = counts.get("VERB", 0) + counts.get("AUX", 0)
    adj  = counts.get("ADJ", 0)
    adv  = counts.get("ADV", 0)
    pron = counts.get("PRON", 0)
    func = (
        counts.get("DET", 0)
        + counts.get("ADP", 0)
        + counts.get("AUX", 0)
        + counts.get("CCONJ", 0)
        + counts.get("SCONJ", 0)
        + counts.get("PART", 0)
    )

    freqs = np.array(list(counts.values()), dtype=np.float64) / total
    entropy = -np.sum(freqs * np.log2(freqs + 1e-12))

    return {
        "pos_noun_ratio": noun / total,
        "pos_verb_ratio": verb / total,
        "pos_adj_ratio": adj / total,
        "pos_adv_ratio": adv / total,
        "pos_pron_ratio": pron / total,
        "pos_func_ratio": func / total,
        "pos_entropy": float(entropy),
    }


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
    #If no sentences exist, set them to 0
    if sentences:
        sent_lens = [len(s.split()) for s in sentences]
        max_sent_len = max(sent_lens)
        min_sent_len = min(sent_lens)
    else:
        max_sent_len = 0
        min_sent_len = 0

    return {
        # counts
        "num_chars": num_chars,
        "num_words": num_words,
        "num_sentences": num_sentences,

        # word length stats
        "avg_word_len": float(np.mean(word_lengths)),
        "std_word_len": float(np.std(word_lengths)),
        "min_word_len": float(np.min(word_lengths)),
        "max_word_len": float(np.max(word_lengths)),

        # sentence length stats
        "avg_sentence_len": float(num_words / num_sentences),
        "max_sentence_len": float(max_sent_len),
        "min_sentence_len": float(min_sent_len),

        # vocabulary richness
        "vocab_size": float(vocab_size),
        "type_token_ratio": float(vocab_size / num_words) if num_words else 0.0,
        "hapax_ratio": float(sum(1 for w, c in counts.items() if c == 1) / num_words) if num_words else 0.0,

        # casing
        "uppercase_ratio": float(sum(1 for c in text if c.isupper()) / max(num_chars, 1)),

        # punctuation
        "punct_ratio": float(len(punct) / max(num_chars, 1)),
        "comma_ratio": float(text.count(",") / max(num_chars, 1)),
        "period_ratio": float(text.count(".") / max(num_chars, 1)),
        "exclamation_ratio": float(text.count("!") / max(num_chars, 1)),
        "question_ratio": float(text.count("?") / max(num_chars, 1)),

        # digits/whitespace
        "digit_ratio": float(sum(1 for c in text if c.isdigit()) / max(num_chars, 1)),
        "whitespace_ratio": float(text.count(" ") / max(num_chars, 1)),
    }

def extract_function_word_features(text):
    words = re.findall(r"\b\w+\b", text.lower())
    num_words = len(words) if words else 1
    counts = Counter(words)

    feats = {}
    for fw in FUNCTION_WORDS:
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
    #handle empty text 
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

def build_feature_matrix(texts):
    feature_dicts = []

    docs = nlp.pipe(texts, batch_size=64)

    for text, doc in zip(texts, docs):
        feats = {}
        feats.update(extract_lexical_features(text))
        feats.update(extract_function_word_features(text))
        feats.update(extract_structure_features(text))
        feats.update(extract_statistical_features(text))
        feats.update(extract_agreement_score(text))
        feats.update(extract_repetition_features(text))
        feats.update(extract_pos_features_doc(doc))  

        feature_dicts.append(feats)

    return pd.DataFrame(feature_dicts).fillna(0.0)


def main():
    full_df = pd.read_csv(TRAIN_PATH)

    #Basic sanity
    if "generation" not in full_df.columns or "model" not in full_df.columns:
        raise ValueError("Expected columns: generation, model. Check the CSV file.")

    print("\nDATASET SIZE INFO")
    print("Full dataset size:", len(full_df))

    #split labeled data into train / test
    train_df, test_df = train_test_split(
        full_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=full_df["model"]
    )

    print("Train size BEFORE filtering:", len(train_df))
    print("Test size BEFORE filtering:", len(test_df))

    """     
    if TOP_K_MODELS is not None:
        top_models = train_df["model"].value_counts().head(TOP_K_MODELS).index.tolist()
        train_df = train_df[train_df["model"].isin(top_models)]
        test_df = test_df[test_df["model"].isin(top_models)]
        print("\nUsing top models:", top_models) 
    """
    if MODELS is not None:
        train_df = train_df[train_df["model"].isin(MODELS)]
        test_df = test_df[test_df["model"].isin(MODELS)]
        print("\nUsing models:", MODELS)

    print("Train size AFTER filtering:", len(train_df))
    print("Test size AFTER filtering:", len(test_df))

    train_texts = train_df["generation"].astype(str)
    train_labels = train_df["model"]

    test_texts = test_df["generation"].astype(str)
    test_labels = test_df["model"]

    print("\nFINAL TRAIN label distribution:")
    print(train_labels.value_counts())

    print("\nFINAL TEST label distribution:")
    print(test_labels.value_counts())

    print("\nBuilding feature matrices")
    X_train = build_feature_matrix(train_texts)
    X_test = build_feature_matrix(test_texts)

    sample_feat_count = X_train.shape[1]
    print("Feature count:", sample_feat_count)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    #Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_labels)
    y_test = le.transform(test_labels)

    print("\nClasses:")
    print(list(le.classes_))

    #Random Forest 
    print("\nTraining RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    print("\nRandomForest Accuracy:", accuracy_score(y_test, rf_preds))
    # Macro F1 = average F1 across all classes (each class weighted equally)
    #gives each class equal importance when averaging their scores.
    print("RandomForest Macro F1:", f1_score(y_test, rf_preds, average="macro"))
    print("RandomForest Confusion Matrix:\n", confusion_matrix(y_test, rf_preds))

    # XGBoost
    print("\nTraining XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        random_state=RANDOM_STATE
    )
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)

    print("\nXGB Accuracy:", accuracy_score(y_test, xgb_preds))
    print("XGB Macro F1:", f1_score(y_test, xgb_preds, average="macro"))
    print("XGB Confusion Matrix:\n", confusion_matrix(y_test, xgb_preds))

    # LightGBM
    print("\nTraining LightGBM...")
    lgbm = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=31,
        random_state=RANDOM_STATE
    )
    lgbm.fit(X_train, y_train)
    lgb_preds = lgbm.predict(X_test)

    print("\nLGBM Accuracy:", accuracy_score(y_test, lgb_preds))
    print("LGBM Macro F1:", f1_score(y_test, lgb_preds, average="macro"))
    print("LGBM Confusion Matrix:\n", confusion_matrix(y_test, lgb_preds))

    # Error analysis
    print("\nError analysis (XGB):")
    errors = test_df.copy()
    errors["true_model"] = test_labels.values
    errors["pred_model"] = le.inverse_transform(xgb_preds)

    mistakes = errors[errors["true_model"] != errors["pred_model"]]
    print("Number of errors:", len(mistakes))

    # Show most common confusions + domain
    top_confusions = mistakes[["true_model", "pred_model", "domain"]].value_counts().head(10)
    print("\nTop confusions:")
    print(top_confusions)

    # SHAP 
    if RUN_SHAP:
        print("\nRunning SHAP ")
        import shap
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test)

    print("\nDone.")


if __name__ == "__main__":
    main()
