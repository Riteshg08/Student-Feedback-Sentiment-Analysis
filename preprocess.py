"""Text preprocessing shared by training and inference."""

from __future__ import annotations

import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from config import NEGATION_SKIP, NEGATION_WORDS

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_TOKEN_RE = re.compile(r"n't|[a-z0-9]+")
_NLTK_PACKAGES = (
    "punkt",
    "punkt_tab",
    "wordnet",
    "omw-1.4",
    "stopwords",
)


def ensure_nltk_data() -> None:
    """Download required NLTK corpora if they are not already present."""
    for package in _NLTK_PACKAGES:
        try:
            if package in {"punkt", "punkt_tab"}:
                nltk.data.find(f"tokenizers/{package}")
            else:
                nltk.data.find(f"corpora/{package}")
        except LookupError:
            nltk.download(package, quiet=True)


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    ensure_nltk_data()
    words = set(stopwords.words("english"))
    words -= set(NEGATION_WORDS)
    words.discard("don")
    return frozenset(words)


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    ensure_nltk_data()
    return WordNetLemmatizer()


def tokenize(text: str) -> list[str]:
    """Lowercase, strip URLs and punctuation, then tokenize."""
    lowered = text.lower()
    lowered = _URL_RE.sub(" ", lowered)
    lowered = lowered.replace("n't", " n't ")
    return _TOKEN_RE.findall(lowered)


def attach_negations(tokens: list[str]) -> list[str]:
    """Join a negation with the next content word, e.g. ``not good`` -> ``not_good``."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        is_neg = token in NEGATION_WORDS or token == "n't" or token.endswith("n't")
        if is_neg:
            j = i + 1
            while j < len(tokens) and tokens[j] in NEGATION_SKIP:
                j += 1
            if j < len(tokens):
                out.append(f"not_{tokens[j]}")
                i = j + 1
                continue
            out.append("not")
            i += 1
            continue
        out.append(token)
        i += 1
    return out


def _lemma_word(lemma: WordNetLemmatizer, token: str) -> str:
    """Lemmatize a token without turning adjectives such as ``boring`` into ``bore``."""
    noun = lemma.lemmatize(token)
    if token.endswith("ing") and noun == token:
        return noun
    verb = lemma.lemmatize(token, pos="v")
    return lemma.lemmatize(verb)


def clean_tokens(tokens: list[str]) -> list[str]:
    """Attach negations first, then lemmatize and drop stopwords (keeping negations)."""
    lemma = _lemmatizer()
    stops = _stopwords()
    cleaned: list[str] = []
    for token in attach_negations(tokens):
        if token.startswith("not_"):
            base = _lemma_word(lemma, token[4:])
            cleaned.append(f"not_{base}" if base else "not")
            continue
        if token in NEGATION_WORDS or token == "n't":
            cleaned.append("not")
            continue
        word = _lemma_word(lemma, token)
        if word in stops or not word:
            continue
        cleaned.append(word)
    return cleaned


def clean_text(text: str | None) -> str:
    """Return a space-joined string of processed tokens.

    The same function is used for uploaded comments and (after joining) the
    JSON training documents, so the vectorizer sees a consistent token space.
    """
    if text is None:
        return ""
    raw = str(text).strip()
    if not raw:
        return ""
    return " ".join(clean_tokens(tokenize(raw)))


def tokens_to_text(tokens: list[str]) -> str:
    """Re-process an already-tokenised training document."""
    return " ".join(clean_tokens([t.lower() for t in tokens if t]))
