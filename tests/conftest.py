from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when pytest is invoked directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from app import create_app
from faculty import load_models
from preprocess import ensure_nltk_data


@pytest.fixture(scope="session")
def app():
    ensure_nltk_data()
    application = create_app(testing=True)
    load_models(train_if_missing=True)
    return application


@pytest.fixture()
def client(app):
    return app.test_client()
