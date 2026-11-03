"""Shared configuration for the Student Feedback Sentiment Analysis app."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
SAMPLES_DIR = BASE_DIR / "samples"
UPLOADS_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR

MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2 MB
ALLOWED_EXTENSIONS = {"csv"}
PII_HEADER_TOKENS = ("name", "email", "e-mail", "email-id", "timestamp")

LABELS: tuple[str, ...] = ("positive", "neutral", "negative")
LOW_CONFIDENCE_THRESHOLD = 0.5
SHORT_TEXT_TOKEN_LIMIT = 4

NEGATION_WORDS = frozenset({"not", "no", "never", "nor", "n't"})
NEGATION_SKIP = frozenset(
    {"so", "very", "really", "too", "that", "quite", "much", "at", "all", "even"}
)

POSITIVE_LEXICON = frozenset(
    {
        "good",
        "great",
        "excellent",
        "awesome",
        "nice",
        "perfect",
        "best",
        "amazing",
        "wonderful",
        "helpful",
        "supportive",
        "clear",
        "punctual",
        "satisfied",
        "satisfactory",
        "love",
        "fantastic",
        "outstanding",
        "superb",
        "well",
        "fine",
        "gud",
        "accurate",
        "interactive",
        "effective",
    }
)
NEGATIVE_LEXICON = frozenset(
    {
        "bad",
        "poor",
        "terrible",
        "worst",
        "pathetic",
        "useless",
        "boring",
        "dull",
        "lazy",
        "horrible",
        "awful",
        "outdated",
        "faulty",
        "inadequate",
        "insufficient",
        "unhelpful",
        "arrogant",
        "bias",
        "partiality",
        "hate",
        "waste",
        "difficult",
        "hard",
        "worse",
        "lacking",
        "unavailable",
        "nonexistent",
        "never",
    }
)
NEUTRAL_LEXICON = frozenset({"ok", "okay", "average", "normal", "fine", "moderate"})


class CategoryConfig(TypedDict):
    id: str
    name: str
    json_file: str
    keywords: list[str]


CATEGORIES: list[CategoryConfig] = [
    {
        "id": "teacher",
        "name": "Teacher Feedback",
        "json_file": "feedback1.json",
        "keywords": [
            "teacher feedback",
            "teacher",
            "faculty",
            "teaching",
            "lecture delivery",
            "punctuality",
        ],
    },
    {
        "id": "course",
        "name": "Course Content",
        "json_file": "feedback2.json",
        "keywords": [
            "course content",
            "course",
            "curriculum",
            "syllabus",
            "syllabi",
            "contents",
        ],
    },
    {
        "id": "exam",
        "name": "Examination Pattern",
        "json_file": "feedback3.json",
        "keywords": [
            "examination pattern",
            "exam pattern",
            "examination",
            "exam",
            "question paper",
            "question",
        ],
    },
    {
        "id": "lab",
        "name": "Laboratory",
        "json_file": "feedback4.json",
        "keywords": ["laboratory", "laboratories", "lab facility", "lab"],
    },
    {
        "id": "library",
        "name": "Library Facilities",
        "json_file": "feedback5.json",
        "keywords": ["library facilities", "library", "books availability"],
    },
    {
        "id": "extra",
        "name": "Extra Co-Curricular Activities",
        "json_file": "feedback6.json",
        "keywords": [
            "extra co-curricular",
            "co-curricular",
            "extracurricular",
            "extra curricular",
            "curricular activities",
            "activities",
        ],
    },
]

SUGGESTIONS = {
    "id": "suggestions",
    "name": "Any other suggestion",
    "keywords": ["any other suggestion", "suggestion", "other comment", "remarks"],
}

CATEGORY_BY_ID = {c["id"]: c for c in CATEGORIES}
CATEGORY_IDS = [c["id"] for c in CATEGORIES]
