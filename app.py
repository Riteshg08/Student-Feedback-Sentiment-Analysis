"""Flask application for student feedback sentiment analysis."""

from __future__ import annotations

import io
import logging
import os
import uuid
from collections import Counter
from typing import Any

import pandas as pd
from flask import Flask, abort, render_template, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from config import (
    CATEGORIES,
    CATEGORY_BY_ID,
    CATEGORY_IDS,
    LABELS,
    MAX_CONTENT_LENGTH,
    PII_HEADER_TOKENS,
    SAMPLES_DIR,
    SUGGESTIONS,
    UPLOADS_DIR,
)
from faculty import Prediction, load_models, predict_many
from preprocess import ensure_nltk_data

logger = logging.getLogger(__name__)


class AnalyseError(Exception):
    """User-facing analysis error that should render as HTTP 400."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


def create_app(*, testing: bool = False) -> Flask:
    app = Flask(__name__)
    app.config["TESTING"] = testing
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or os.urandom(32)
    debug_flag = os.environ.get("FLASK_DEBUG", os.environ.get("DEBUG", "0")).lower()
    app.config["DEBUG"] = False if testing else debug_flag in {"1", "true", "yes"}

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    @app.errorhandler(AnalyseError)
    def handle_analyse_error(exc: AnalyseError):
        return (
            render_template("error.html", message=exc.message, hint=exc.hint),
            400,
        )

    @app.errorhandler(RequestEntityTooLarge)
    def handle_too_large(_exc: RequestEntityTooLarge):
        return (
            render_template(
                "error.html",
                message="The file is too large.",
                hint="Please upload a CSV smaller than 2 MB.",
            ),
            400,
        )

    @app.errorhandler(400)
    def handle_400(exc):
        message = getattr(exc, "description", None) or "The request could not be processed."
        return render_template("error.html", message=message, hint=""), 400

    @app.route("/")
    def display():
        return render_template("index.html", categories=CATEGORIES)

    @app.route("/samples/<path:filename>")
    def sample_file(filename: str):
        safe = secure_filename(filename)
        path = SAMPLES_DIR / safe
        if not path.is_file():
            abort(404)
        return send_from_directory(SAMPLES_DIR, safe, as_attachment=False)

    @app.route("/", methods=["POST"])
    def analyse_upload():
        try:
            mapping = _mapping_from_form(request.form)
            if request.form.get("upload_id"):
                df = _load_pending_upload(request.form.get("upload_id", ""))
                selected = parse_category_ids(request.form.get("Cate", "all"))
            else:
                df = parse_csv(request)
                selected = parse_category_ids(request.form.get("Cate", "all"))
                auto = match_columns(df, selected)
                if mapping:
                    auto.update(mapping)
                missing = [cid for cid in selected if cid not in auto]
                if missing:
                    upload_id = _stash_dataframe(df)
                    return render_template(
                        "map_columns.html",
                        categories=[CATEGORY_BY_ID[c] for c in selected],
                        matched=auto,
                        missing=missing,
                        columns=list(df.columns),
                        upload_id=upload_id,
                        cate=request.form.get("Cate", "all"),
                    )
                mapping = auto

            results = build_results(df, selected, mapping)
            _cleanup_upload(request.form.get("upload_id"))
            return render_template("results.html", results=results)
        except AnalyseError:
            raise
        except Exception:
            logger.exception("Unexpected error while analysing upload")
            raise AnalyseError(
                "Something went wrong while analysing this file.",
                "Check that the file is a UTF-8 CSV exported from Google Forms, then try again.",
            )

    if not testing:
        ensure_nltk_data()
        load_models(train_if_missing=True)

    return app


def parse_category_ids(value: str | None) -> list[str]:
    """Parse the category selector. ``all`` / ``0`` / empty means every category."""
    raw = (value or "all").strip()
    if raw.lower() in {"all", "0", ""}:
        return list(CATEGORY_IDS)
    if raw not in CATEGORY_BY_ID:
        raise AnalyseError(
            "The category selection is not valid.",
            "Choose All categories or one of the listed categories. Cate must not be a random value.",
        )
    return [raw]


def parse_csv(req) -> pd.DataFrame:
    """Read an uploaded CSV entirely in memory. Never writes the upload to disk."""
    if "file_name" not in req.files:
        raise AnalyseError(
            "No file was uploaded.",
            "Choose a CSV file exported from Google Forms and submit the form again.",
        )
    upload = req.files["file_name"]
    filename = upload.filename or ""
    if not filename.strip():
        raise AnalyseError(
            "No file was uploaded.",
            "Choose a CSV file exported from Google Forms and submit the form again.",
        )
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix != "csv":
        raise AnalyseError(
            "That file does not look like a CSV.",
            "Export the Google Form responses as a .csv file (UTF-8) and try again.",
        )

    raw = upload.read()
    if not raw or not raw.strip():
        raise AnalyseError(
            "The uploaded file is empty.",
            "Export the form responses again and make sure the CSV contains a header row and at least one response.",
        )

    text = _decode_csv_bytes(raw)
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        logger.info("CSV parse failed: %s", exc)
        raise AnalyseError(
            "The CSV could not be read.",
            "Save the file as UTF-8 CSV. Merged cells, extra header rows, or a non-CSV format will fail.",
        ) from exc

    if df.empty or len(df.columns) == 0:
        raise AnalyseError(
            "The uploaded file is empty.",
            "The CSV needs a header row and at least one student response.",
        )
    if len(df.columns) <= 2:
        raise AnalyseError(
            "This CSV does not have enough columns.",
            "A feedback export should include one column per question. Two or fewer columns is not a valid form dump.",
        )
    return df


def _decode_csv_bytes(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise AnalyseError(
        "The file is not valid UTF-8.",
        "Re-export the Google Form as a CSV (File → Download → CSV) so the text encoding is UTF-8.",
    )


def is_pii_column(name: str) -> bool:
    lowered = name.strip().lower()
    return any(token == lowered or token in lowered.split() for token in PII_HEADER_TOKENS) or any(
        token in lowered for token in ("email", "timestamp")
    )


def usable_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not is_pii_column(str(c))]


def _keyword_score(column: str, keywords: list[str]) -> int:
    hay = column.strip().lower()
    score = 0
    for keyword in keywords:
        key = keyword.lower()
        if hay == key:
            score += 50 + len(key)
        elif key in hay:
            score += 10 + len(key)
    return score


def match_columns(df: pd.DataFrame, selected: list[str]) -> dict[str, str]:
    """Match each selected category to the best unused CSV column."""
    remaining = usable_columns(df)
    matched: dict[str, str] = {}
    for cat in CATEGORIES:
        if cat["id"] not in selected:
            continue
        scored: list[tuple[int, str]] = []
        for col in remaining:
            score = _keyword_score(str(col), cat["keywords"])
            if score > 0:
                scored.append((score, col))
        if not scored:
            continue
        scored.sort(key=lambda item: (-item[0], remaining.index(item[1])))
        winner = scored[0][1]
        matched[cat["id"]] = winner
        remaining = [c for c in remaining if c != winner]
    return matched


def match_suggestions_column(df: pd.DataFrame, used: set[str]) -> str | None:
    remaining = [c for c in usable_columns(df) if c not in used]
    scored = []
    for col in remaining:
        score = _keyword_score(str(col), list(SUGGESTIONS["keywords"]))
        if score > 0:
            scored.append((score, col))
    if not scored:
        return None
    scored.sort(key=lambda item: -item[0])
    return scored[0][1]


def _mapping_from_form(form) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for key, value in form.items():
        if key.startswith("map_") and value.strip():
            cid = key[4:]
            if cid in CATEGORY_BY_ID:
                mapping[cid] = value.strip()
    return mapping


def _stash_dataframe(df: pd.DataFrame) -> str:
    """Persist a CSV only when a second mapping step is required."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    upload_id = uuid.uuid4().hex
    path = UPLOADS_DIR / f"{upload_id}.csv"
    df.to_csv(path, index=False)
    return upload_id


def _load_pending_upload(upload_id: str) -> pd.DataFrame:
    safe = secure_filename(upload_id)
    path = UPLOADS_DIR / f"{safe}.csv"
    if not path.is_file():
        raise AnalyseError(
            "The uploaded file expired.",
            "Please upload the CSV again and complete the column mapping in one go.",
        )
    return pd.read_csv(path)


def _cleanup_upload(upload_id: str | None) -> None:
    if not upload_id:
        return
    path = UPLOADS_DIR / f"{secure_filename(upload_id)}.csv"
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        logger.warning("Could not delete temporary upload %s", path)


def non_blank_values(series: pd.Series) -> tuple[list[str], int]:
    values: list[str] = []
    blanks = 0
    for item in series.tolist():
        if item is None or (isinstance(item, float) and pd.isna(item)):
            blanks += 1
            continue
        text = str(item).strip()
        if not text or text.lower() in {"nan", "none"}:
            blanks += 1
            continue
        values.append(str(item).strip())
    return values, blanks


def top_keywords(predictions: list[Prediction], label: str, limit: int = 8) -> list[str]:
    counts: Counter[str] = Counter()
    for pred in predictions:
        if pred.label != label:
            continue
        counts.update(tok for tok in pred.cleaned.split() if len(tok) > 2)
    return [word for word, _ in counts.most_common(limit)]


def example_comments(predictions: list[Prediction], limit: int = 3) -> list[dict[str, Any]]:
    picked: list[Prediction] = []
    for label in LABELS:
        for pred in predictions:
            if pred.label == label and pred not in picked:
                picked.append(pred)
                break
    if len(picked) < limit:
        for pred in predictions:
            if pred not in picked:
                picked.append(pred)
            if len(picked) >= limit:
                break
    examples = []
    for pred in picked[:limit]:
        snippet = pred.text.replace("\n", " ").strip()
        if len(snippet) > 180:
            snippet = snippet[:177] + "…"
        examples.append(
            {
                "text": snippet,
                "label": pred.label,
                "confidence": round(pred.confidence, 2),
                "uncertain": pred.uncertain,
            }
        )
    return examples


def analyse_column(series: pd.Series, category_id: str) -> dict[str, Any]:
    texts, blanks = non_blank_values(series)
    predictions = predict_many(texts, category_id)
    counts = Counter(p.label for p in predictions)
    total = len(predictions)
    percents = {
        label: round(100.0 * counts.get(label, 0) / total, 1) if total else 0.0 for label in LABELS
    }
    return {
        "id": category_id,
        "name": CATEGORY_BY_ID[category_id]["name"],
        "column": str(series.name),
        "positive": int(counts.get("positive", 0)),
        "neutral": int(counts.get("neutral", 0)),
        "negative": int(counts.get("negative", 0)),
        "blanks": blanks,
        "total": total,
        "percent": percents,
        "keywords": {label: top_keywords(predictions, label) for label in LABELS},
        "examples": example_comments(predictions),
        "predictions": [
            {
                "text": p.text.replace("\n", " ").strip()[:240],
                "label": p.label,
                "confidence": round(p.confidence, 2),
                "uncertain": p.uncertain,
                "source": p.source,
            }
            for p in predictions
        ],
    }


def analyse_suggestions(series: pd.Series) -> dict[str, Any]:
    texts, blanks = non_blank_values(series)
    predictions = predict_many(texts, CATEGORY_IDS[0])
    return {
        "name": SUGGESTIONS["name"],
        "column": str(series.name),
        "blanks": blanks,
        "total": len(predictions),
        "comments": [
            {
                "text": p.text.replace("\n", " ").strip()[:400],
                "label": p.label,
                "confidence": round(p.confidence, 2),
                "uncertain": p.uncertain,
            }
            for p in predictions
        ],
        "counts": {
            "positive": sum(1 for p in predictions if p.label == "positive"),
            "neutral": sum(1 for p in predictions if p.label == "neutral"),
            "negative": sum(1 for p in predictions if p.label == "negative"),
        },
    }


def build_results(
    df: pd.DataFrame, selected: list[str], mapping: dict[str, str]
) -> dict[str, Any]:
    if len(df.columns) < len(mapping):
        raise AnalyseError(
            "This CSV has fewer columns than the categories you selected.",
            "Pick fewer categories, or upload a full Google Forms export.",
        )

    categories_out: list[dict[str, Any]] = []
    total_blanks = 0
    total_answers = 0
    for cid in selected:
        column = mapping.get(cid)
        if not column:
            continue
        if column not in df.columns:
            raise AnalyseError(
                f"The mapped column “{column}” is not in this CSV.",
                "Go back and choose an existing column for each category.",
            )
        result = analyse_column(df[column], cid)
        categories_out.append(result)
        total_blanks += result["blanks"]
        total_answers += result["total"]

    if not categories_out:
        raise AnalyseError(
            "None of the selected categories could be matched to a column.",
            "Use the mapping form to choose which CSV column belongs to each category.",
        )

    used = {c["column"] for c in categories_out}
    suggestion_col = match_suggestions_column(df, used)
    suggestions = analyse_suggestions(df[suggestion_col]) if suggestion_col else None

    pos = sum(c["positive"] for c in categories_out)
    neu = sum(c["neutral"] for c in categories_out)
    neg = sum(c["negative"] for c in categories_out)
    scored = pos + neu + neg
    overall_score = round(100.0 * (pos - neg) / scored, 1) if scored else 0.0

    needs_attention = sorted(
        [
            {
                "id": c["id"],
                "name": c["name"],
                "negative": c["negative"],
                "percent": c["percent"]["negative"],
                "total": c["total"],
            }
            for c in categories_out
            if c["total"] and c["percent"]["negative"] >= 25
        ],
        key=lambda row: (-row["percent"], -row["negative"]),
    )
    if not needs_attention:
        worst = max(categories_out, key=lambda c: c["percent"]["negative"] if c["total"] else -1)
        if worst["total"]:
            needs_attention = [
                {
                    "id": worst["id"],
                    "name": worst["name"],
                    "negative": worst["negative"],
                    "percent": worst["percent"]["negative"],
                    "total": worst["total"],
                }
            ]

    return {
        "row_count": int(len(df)),
        "total_answers": total_answers,
        "blank_skipped": total_blanks,
        "overall_score": overall_score,
        "overall_counts": {"positive": pos, "neutral": neu, "negative": neg},
        "categories": categories_out,
        "suggestions": suggestions,
        "needs_attention": needs_attention,
        "labels": list(LABELS),
    }


app = create_app()


if __name__ == "__main__":
    ensure_nltk_data()
    load_models(train_if_missing=True)
    debug = app.config["DEBUG"]
    app.run(debug=debug, host="127.0.0.1", port=int(os.environ.get("PORT", "5000")))
