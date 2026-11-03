"""Load saved category models and classify student comments."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from config import (
    CATEGORIES,
    CATEGORY_BY_ID,
    LABELS,
    LOW_CONFIDENCE_THRESHOLD,
    MODELS_DIR,
    NEGATIVE_LEXICON,
    NEUTRAL_LEXICON,
    POSITIVE_LEXICON,
    SHORT_TEXT_TOKEN_LIMIT,
)
from preprocess import clean_text, ensure_nltk_data
from train import train_all

logger = logging.getLogger(__name__)

_MODELS: dict[str, Pipeline] = {}


@dataclass
class Prediction:
    text: str
    cleaned: str
    label: str
    confidence: float
    uncertain: bool
    source: str  # "model" | "lexicon" | "hybrid"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def lexicon_score(cleaned: str) -> tuple[str | None, float, float]:
    """Return (label or None, signed score, confidence) from a small lexicon.

    Negated tokens such as ``not_good`` flip the polarity of the attached word.
    """
    if not cleaned:
        return None, 0.0, 0.0
    score = 0.0
    hits = 0
    for token in cleaned.split():
        polarity = 0.0
        if token.startswith("not_"):
            base = token[4:]
            if base in NEGATIVE_LEXICON:
                polarity = 1.2
            elif base in POSITIVE_LEXICON:
                polarity = -1.4
            elif base in NEUTRAL_LEXICON:
                polarity = -0.4
        elif token in NEGATIVE_LEXICON:
            polarity = -1.0
        elif token in POSITIVE_LEXICON:
            polarity = 1.0
        elif token in NEUTRAL_LEXICON:
            polarity = 0.15
        if polarity:
            score += polarity
            hits += 1
    if not hits:
        return None, 0.0, 0.0
    if score > 0.35:
        label = "positive"
    elif score < -0.35:
        label = "negative"
    else:
        label = "neutral"
    confidence = min(0.99, 0.55 + 0.12 * hits + 0.08 * abs(score))
    return label, score, confidence


def _model_predict(pipeline: Pipeline, cleaned: str) -> tuple[str, float]:
    label = str(pipeline.predict([cleaned])[0])
    confidence = 0.33
    clf = pipeline.named_steps.get("clf")
    if hasattr(clf, "predict_proba"):
        proba = pipeline.predict_proba([cleaned])[0]
        classes = list(pipeline.classes_)
        confidence = float(max(proba))
        label = str(classes[int(proba.argmax())])
    elif hasattr(clf, "decision_function"):
        scores = pipeline.decision_function([cleaned])
        raw = scores[0]
        if getattr(raw, "ndim", 0) == 0:
            # Binary decision: map to the two classes.
            confidence = float(abs(raw))
            confidence = 1 / (1 + pow(2.718281828, -confidence))
        else:
            shifted = raw - np.max(raw)
            exp = np.exp(shifted)
            proba = exp / exp.sum()
            confidence = float(max(proba))
            label = str(pipeline.classes_[int(proba.argmax())])
    return label, confidence


def predict_one(text: str, pipeline: Pipeline) -> Prediction:
    """Classify a single comment with lexicon fallback for short / low-confidence text."""
    cleaned = clean_text(text)
    lex_label, _lex_score, lex_conf = lexicon_score(cleaned)
    tokens = cleaned.split()
    short = len(tokens) <= SHORT_TEXT_TOKEN_LIMIT

    if not cleaned:
        return Prediction(
            text=text,
            cleaned=cleaned,
            label="neutral",
            confidence=0.0,
            uncertain=True,
            source="lexicon",
        )

    model_label, model_conf = _model_predict(pipeline, cleaned)

    chosen_label = model_label
    chosen_conf = model_conf
    source = "model"

    if short and lex_label is not None and lex_conf >= 0.6:
        chosen_label, chosen_conf, source = lex_label, max(lex_conf, model_conf), "lexicon"
    elif model_conf < LOW_CONFIDENCE_THRESHOLD and lex_label is not None and lex_conf >= 0.65:
        chosen_label, chosen_conf, source = lex_label, lex_conf, "hybrid"

    uncertain = chosen_conf < LOW_CONFIDENCE_THRESHOLD
    return Prediction(
        text=str(text),
        cleaned=cleaned,
        label=chosen_label,
        confidence=float(chosen_conf),
        uncertain=uncertain,
        source=source,
    )


def predict_many(texts: Iterable[str], category_id: str) -> list[Prediction]:
    pipeline = get_model(category_id)
    return [predict_one(text, pipeline) for text in texts]


def models_ready() -> bool:
    return all((MODELS_DIR / f"{cat['id']}.joblib").exists() for cat in CATEGORIES)


def load_models(train_if_missing: bool = True) -> dict[str, Pipeline]:
    """Load joblib pipelines once. Train them if ``models/`` is incomplete."""
    global _MODELS
    if _MODELS and all(cid in _MODELS for cid in CATEGORY_BY_ID):
        return _MODELS
    ensure_nltk_data()
    if train_if_missing and not models_ready():
        logger.info("Saved models missing; training now.")
        train_all()
    loaded: dict[str, Pipeline] = {}
    for cat in CATEGORIES:
        path = MODELS_DIR / f"{cat['id']}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model not found for {cat['id']}: {path}")
        loaded[cat["id"]] = joblib.load(path)
    _MODELS = loaded
    return _MODELS


def get_model(category_id: str) -> Pipeline:
    if category_id not in _MODELS:
        load_models()
    return _MODELS[category_id]


def predict_label(text: str, category_id: str = "teacher") -> Prediction:
    """Public helper used by tests and the CLI."""
    return predict_one(text, get_model(category_id))
