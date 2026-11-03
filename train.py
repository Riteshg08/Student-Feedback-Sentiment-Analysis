"""Train one sentiment pipeline per feedback category and save metrics."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import BASE_DIR, CATEGORIES, LABELS, MODELS_DIR
from preprocess import ensure_nltk_data, tokens_to_text

RANDOM_STATE = 42


def load_documents(json_path: Path) -> list[tuple[list[str], str]]:
    """Load ``[tokens, label]`` pairs from a training JSON file."""
    with json_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    docs: list[tuple[list[str], str]] = []
    for item in raw:
        tokens, label = item[0], str(item[1]).strip().lower()
        if label not in LABELS:
            continue
        docs.append(([str(t) for t in tokens], label))
    return docs


def deduplicate(docs: list[tuple[list[str], str]]) -> list[tuple[list[str], str]]:
    """Drop duplicate (token-sequence, label) pairs, keeping first occurrence."""
    seen: set[tuple[tuple[str, ...], str]] = set()
    unique: list[tuple[list[str], str]] = []
    for tokens, label in docs:
        key = (tuple(tokens), label)
        if key in seen:
            continue
        seen.add(key)
        unique.append((tokens, label))
    return unique


def make_pipeline(estimator: Any) -> Pipeline:
    """TF-IDF unigrams+bigrams followed by a classifier."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            ("clf", estimator),
        ]
    )


def candidate_estimators() -> dict[str, Any]:
    return {
        "logreg": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "linearsvc": LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
        "nb": MultinomialNB(),
    }


def evaluate_estimator(
    name: str,
    estimator: Any,
    texts: list[str],
    labels: list[str],
    n_splits: int,
) -> dict[str, Any]:
    pipeline = make_pipeline(estimator)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    preds = cross_val_predict(pipeline, texts, labels, cv=cv)
    acc = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro", labels=list(LABELS), zero_division=0))
    report = classification_report(
        labels, preds, labels=list(LABELS), output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(labels, preds, labels=list(LABELS)).tolist()
    fold_sizes = []
    for train_idx, test_idx in cv.split(texts, labels):
        fold_sizes.append({"train": int(len(train_idx)), "test": int(len(test_idx))})
    return {
        "name": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "report": report,
        "confusion_matrix": matrix,
        "fold_sizes": fold_sizes,
        "predictions": preds,
    }


def majority_baseline(labels: list[str]) -> dict[str, Any]:
    dummy = DummyClassifier(strategy="most_frequent")
    X = np.zeros((len(labels), 1))
    dummy.fit(X, labels)
    preds = dummy.predict(X)
    return {
        "name": "majority",
        "majority_class": Counter(labels).most_common(1)[0][0],
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(
            f1_score(labels, preds, average="macro", labels=list(LABELS), zero_division=0)
        ),
    }


def choose_n_splits(labels: list[str]) -> int:
    counts = Counter(labels)
    min_count = min(counts.values()) if counts else 0
    return max(2, min(5, min_count))


def train_category(category: dict[str, Any]) -> dict[str, Any]:
    json_path = BASE_DIR / category["json_file"]
    original = load_documents(json_path)
    unique = deduplicate(original)
    texts = [tokens_to_text(tokens) for tokens, _ in unique]
    labels = [label for _, label in unique]

    # Drop empty documents (all-stopword rows).
    paired = [(t, y) for t, y in zip(texts, labels) if t.strip()]
    texts = [t for t, _ in paired]
    labels = [y for _, y in paired]

    n_splits = choose_n_splits(labels)
    results = [
        evaluate_estimator(name, est, texts, labels, n_splits)
        for name, est in candidate_estimators().items()
    ]
    best = max(results, key=lambda row: (row["macro_f1"], row["accuracy"]))
    winner = candidate_estimators()[best["name"]]
    pipeline = make_pipeline(winner)
    pipeline.fit(texts, labels)
    model_path = MODELS_DIR / f"{category['id']}.joblib"
    joblib.dump(pipeline, model_path)

    baseline = majority_baseline(labels)
    label_counts = dict(Counter(labels))
    print(f"\n=== {category['name']} ({category['id']}) ===")
    print(f"Original samples: {len(original)}  Unique: {len(unique)}  Usable: {len(texts)}")
    print(f"Class counts: {label_counts}")
    print(f"CV folds: {n_splits}  train/test per fold: {best['fold_sizes']}")
    for row in results:
        print(f"  {row['name']:10}  acc={row['accuracy']:.3f}  macro-F1={row['macro_f1']:.3f}")
    print(
        f"  baseline    acc={baseline['accuracy']:.3f}  macro-F1={baseline['macro_f1']:.3f}"
        f"  (class={baseline['majority_class']})"
    )
    print(f"Selected: {best['name']}")
    print(classification_report(labels, best["predictions"], labels=list(LABELS), zero_division=0))

    return {
        "id": category["id"],
        "name": category["name"],
        "original_count": len(original),
        "unique_count": len(unique),
        "usable_count": len(texts),
        "class_counts": label_counts,
        "n_splits": n_splits,
        "fold_sizes": best["fold_sizes"],
        "candidates": {
            row["name"]: {
                "accuracy": row["accuracy"],
                "macro_f1": row["macro_f1"],
                "report": row["report"],
                "confusion_matrix": row["confusion_matrix"],
            }
            for row in results
        },
        "selected_model": best["name"],
        "accuracy": best["accuracy"],
        "macro_f1": best["macro_f1"],
        "report": best["report"],
        "confusion_matrix": best["confusion_matrix"],
        "baseline": baseline,
        "model_path": str(model_path.relative_to(BASE_DIR)),
    }


def train_all() -> dict[str, Any]:
    ensure_nltk_data()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    categories = [train_category(cat) for cat in CATEGORIES]
    mean_acc = float(np.mean([c["accuracy"] for c in categories]))
    mean_f1 = float(np.mean([c["macro_f1"] for c in categories]))
    metrics = {
        "old_baseline": {
            "model": "CountVectorizer + MultinomialNB (retrained every request)",
            "accuracy": 0.62,
            "macro_f1": 0.55,
            "neutral_recall": 0.21,
        },
        "mean_accuracy": mean_acc,
        "mean_macro_f1": mean_f1,
        "labels": list(LABELS),
        "categories": categories,
    }
    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\n==== Overall ====")
    print(f"Mean accuracy: {mean_acc:.3f}")
    print(f"Mean macro-F1: {mean_f1:.3f}  (old baseline 0.55)")
    print(f"Wrote {metrics_path}")
    return metrics


def main() -> int:
    train_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
