import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from xgboost import XGBClassifier
import lightgbm as lgb


TRAIN_PATH = "raid_data/train_none.csv"

MODELS = ["cohere-chat", "gpt4", "mistral-chat", "mpt-chat", "llama-chat"]
RANDOM_STATE = 5


def build_tfidf_matrix(train_texts, test_texts):
    tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        max_features=2000
    )

    X_train = tfidf.fit_transform(train_texts)
    X_test = tfidf.transform(test_texts)

    print("TF-IDF train shape:", X_train.shape)
    print("TF-IDF test shape:", X_test.shape)

    return X_train, X_test, tfidf


def main():
    df = pd.read_csv(TRAIN_PATH)

    if "generation" not in df.columns or "model" not in df.columns:
        raise ValueError("Expected columns: generation, model")

    print("\nDATASET SIZE INFO")
    print("Full dataset size:", len(df))

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["model"],
        random_state=RANDOM_STATE
    )

    # Filter to selected models
    train_df = train_df[train_df["model"].isin(MODELS)]
    test_df = test_df[test_df["model"].isin(MODELS)]

    print("\nTrain size AFTER filtering:", len(train_df))
    print("Test size AFTER filtering:", len(test_df))

    train_texts = train_df["generation"].astype(str)
    test_texts = test_df["generation"].astype(str)

    print("\nFINAL TRAIN label distribution:")
    print(train_df["model"].value_counts())

    print("\nFINAL TEST label distribution:")
    print(test_df["model"].value_counts())


    print("\nBuilding TF-IDF features...")
    X_train, X_test, tfidf = build_tfidf_matrix(train_texts, test_texts)

    print("TF-IDF feature count:", X_train.shape[1])

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["model"])
    y_test = le.transform(test_df["model"])

    print("\nClasses:")
    print(list(le.classes_))


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
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)

    print("\nXGB Accuracy:", accuracy_score(y_test, xgb_preds))
    print("XGB Macro F1:", f1_score(y_test, xgb_preds, average="macro"))
    print("XGB Confusion Matrix:\n", confusion_matrix(y_test, xgb_preds))


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


    print("\nError analysis (XGB):")
    errors = test_df.copy()
    errors["true_model"] = test_df["model"].values
    errors["pred_model"] = le.inverse_transform(xgb_preds)

    mistakes = errors[errors["true_model"] != errors["pred_model"]]
    print("Number of errors:", len(mistakes))

    if "domain" in mistakes.columns:
        top_confusions = (
            mistakes[["true_model", "pred_model", "domain"]]
            .value_counts()
            .head(10)
        )
        print("\nTop confusions:")
        print(top_confusions)

    print("\nDone.")


if __name__ == "__main__":
    main()
