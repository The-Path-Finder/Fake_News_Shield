"""Offline Model Development Pipeline for Fake News Shield.

Trains and compares Logistic Regression (baseline), Multinomial Naive Bayes,
and Linear SVM using spaCy preprocessing and TF-IDF (unigrams + bigrams) on
English (Kaggle) and Hindi/multilingual (HinFakeNews / Zenodo) datasets.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from preprocessing import preprocess_multilingual


def load_and_prepare_data(
    english_csv: Path | None,
    hindi_csv: Path | None,
) -> pd.DataFrame:
    """Load labelled datasets, drop missing/duplicate rows, and preprocess text."""
    frames = []

    if english_csv and english_csv.exists():
        df_en = pd.read_csv(english_csv)
        frames.append(df_en[["text", "label"]])

    if hindi_csv and hindi_csv.exists():
        df_hi = pd.read_csv(hindi_csv)
        frames.append(df_hi[["text", "label"]])

    if not frames:
        raise FileNotFoundError(
            "No dataset CSVs found. Pass --english-csv and/or --hindi-csv "
            "with 'text' and 'label' (0=Fake, 1=Real) columns."
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["text", "label"])
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df["cleaned_text"] = df["text"].astype(str).apply(
        lambda t: preprocess_multilingual(t)["cleaned_text"]
    )
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)
    return df


def train_and_evaluate(df: pd.DataFrame, output_dir: Path) -> None:
    """Vectorize with TF-IDF (1-2 grams), compare 3 classifiers, and save the best."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(random_state=42),
    }

    best_name = None
    best_model = None
    best_f1 = -1.0

    for name, clf in classifiers.items():
        clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds)

        print(f"\n=== {name} ===")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print(classification_report(y_test, preds, zero_division=0))

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = clf

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, output_dir / "linear_svm_model.pkl")
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.pkl")
    print(f"\nSaved best model ({best_name}, F1={best_f1:.4f}) to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Fake News Shield NLP models.")
    parser.add_argument("--english-csv", type=Path, help="Path to English dataset CSV (text, label)")
    parser.add_argument("--hindi-csv", type=Path, help="Path to Hindi/multilingual dataset CSV (text, label)")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory to save .pkl files")
    args = parser.parse_args()

    df = load_and_prepare_data(args.english_csv, args.hindi_csv)
    train_and_evaluate(df, args.output_dir)


if __name__ == "__main__":
    main()
