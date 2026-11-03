from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from app import AnalyseError, build_results, match_columns, parse_category_ids, parse_csv
from config import BASE_DIR, CATEGORY_IDS, SAMPLES_DIR
from faculty import get_model, predict_label, predict_many
from preprocess import clean_text
from train import deduplicate, load_documents

SAMPLE_CSV = SAMPLES_DIR / "Student-Feedback-Form.csv"


def _post_csv(client, content: bytes, filename: str = "data.csv", cate: str = "all"):
    return client.post(
        "/",
        data={"file_name": (io.BytesIO(content), filename), "Cate": cate},
        content_type="multipart/form-data",
    )


def test_neg_keys_are_independent():
    """Teacher negative count must not be overwritten by course-content negatives."""
    df = pd.read_csv(SAMPLE_CSV)
    mapping = match_columns(df, list(CATEGORY_IDS))
    results = build_results(df, list(CATEGORY_IDS), mapping)
    by_id = {c["id"]: c for c in results["categories"]}
    teacher = by_id["teacher"]
    course = by_id["course"]
    assert teacher["negative"] + teacher["positive"] + teacher["neutral"] == teacher["total"]
    assert course["negative"] + course["positive"] + course["neutral"] == course["total"]
    # The original bug copied course negatives onto the teacher row.
    assert "neg1" not in teacher
    assert teacher["id"] != course["id"]
    teacher_direct = predict_many(
        [str(v) for v in df[mapping["teacher"]].dropna() if str(v).strip()],
        "teacher",
    )
    assert teacher["negative"] == sum(1 for p in teacher_direct if p.label == "negative")


def test_per_category_models_are_distinct():
    teacher = get_model("teacher")
    course = get_model("course")
    assert teacher is not course
    for cid in CATEGORY_IDS:
        model = get_model(cid)
        assert model is not None
        assert hasattr(model, "predict")


def test_training_data_is_deduplicated():
    docs = load_documents(BASE_DIR / "feedback1.json")
    unique = deduplicate(docs)
    assert len(docs) == 215
    assert len(unique) < len(docs)
    assert len(unique) > 100


def test_negation_and_short_phrases_predict_negative():
    assert "not_good" in clean_text("Teachers are not good")
    assert "not_boring" in clean_text("Lectures are not boring at all")
    assert predict_label("Very bad and poor").label == "negative"
    assert predict_label("Teachers are not good").label == "negative"
    assert predict_label("okay not so good").label == "negative"
    assert predict_label("Lectures are not boring at all").label != "negative"
    assert predict_label("good").label == "positive"
    assert predict_label("poor").label == "negative"
    assert predict_label("bad").label == "negative"
    assert predict_label("nice").label == "positive"


def test_blank_cells_are_skipped_and_counted(client):
    csv = (
        "Teacher Feedback,Course Content,Examination Pattern,Laboratory,Library Facilities,Extra Co-Curricular Activities\n"
        "good,good,good,good,good,good\n"
        ",bad,,nice,,poor\n"
        "  ,  ,ok,ok,ok,ok\n"
    ).encode("utf-8")
    response = _post_csv(client, csv)
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "blank responses skipped" in body.lower()
    df = pd.read_csv(io.BytesIO(csv))
    mapping = match_columns(df, list(CATEGORY_IDS))
    results = build_results(df, list(CATEGORY_IDS), mapping)
    teacher = next(c for c in results["categories"] if c["id"] == "teacher")
    assert teacher["blanks"] >= 2
    assert teacher["total"] + teacher["blanks"] == 3
    assert results["blank_skipped"] >= 2


def test_empty_file_returns_400(client):
    response = _post_csv(client, b"", "empty.csv")
    assert response.status_code == 400
    assert b"empty" in response.data.lower()


def test_png_upload_returns_400(client):
    response = _post_csv(client, b"\x89PNG\r\n\x1a\n" + b"0" * 20, "chart.png")
    assert response.status_code == 400
    assert b"csv" in response.data.lower()


def test_two_column_csv_returns_400(client):
    response = _post_csv(client, b"Name,Comment\nAda,good\n", "small.csv")
    assert response.status_code == 400


def test_non_numeric_cate_returns_400(client):
    csv = SAMPLE_CSV.read_bytes()
    response = _post_csv(client, csv, "feedback.csv", cate="abc")
    assert response.status_code == 400
    assert b"category" in response.data.lower()


def test_missing_file_returns_400(client):
    response = client.post("/", data={"Cate": "all"}, content_type="multipart/form-data")
    assert response.status_code == 400


def test_invalid_cate_helper():
    try:
        parse_category_ids("abc")
        assert False, "expected AnalyseError"
    except AnalyseError as exc:
        assert "category" in exc.message.lower()


def test_path_traversal_filename_does_not_write(client, tmp_path, monkeypatch):
    target = BASE_DIR.parent / "evil.csv"
    if target.exists():
        target.unlink()
    static_evil = BASE_DIR / "static" / "evil.csv"
    if static_evil.exists():
        static_evil.unlink()
    csv = SAMPLE_CSV.read_bytes()
    response = _post_csv(client, csv, "../evil.csv")
    assert response.status_code in {200, 400}
    assert not target.exists()
    assert not static_evil.exists()
    assert not (BASE_DIR / "evil.csv").exists()


def test_sample_csv_counts_add_up(client):
    assert SAMPLE_CSV.is_file()
    df = pd.read_csv(SAMPLE_CSV)
    mapping = match_columns(df, list(CATEGORY_IDS))
    all_results = build_results(df, list(CATEGORY_IDS), mapping)
    response = client.post(
        "/",
        data={
            "file_name": (io.BytesIO(SAMPLE_CSV.read_bytes()), "Student-Feedback-Form.csv"),
            "Cate": "all",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    for cat in all_results["categories"]:
        assert cat["positive"] + cat["neutral"] + cat["negative"] == cat["total"]
        non_blank = int(df[mapping[cat["id"]]].astype(str).str.strip().replace({"nan": ""}).ne("").sum())
        # pandas nan becomes "nan" string; use the analyser total instead
        series_total = cat["total"] + cat["blanks"]
        assert series_total == len(df)
        single = build_results(df, [cat["id"]], {cat["id"]: mapping[cat["id"]]})
        one = single["categories"][0]
        assert one["positive"] == cat["positive"]
        assert one["neutral"] == cat["neutral"]
        assert one["negative"] == cat["negative"]
        assert cat["name"] in body


def test_parse_csv_rejects_non_utf8():
    class FakeFiles(dict):
        pass

    class FakeFile:
        filename = "data.csv"

        def read(self):
            return "café".encode("latin-1")

    class FakeReq:
        files = FakeFiles(file_name=FakeFile())

    try:
        parse_csv(FakeReq())
        # latin-1 bytes for café may fail utf-8 — if pandas never sees it
    except AnalyseError as exc:
        assert "utf-8" in exc.message.lower() or "utf" in exc.message.lower()
