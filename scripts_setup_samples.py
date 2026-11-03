"""One-off helper: anonymise sample CSVs and tidy legacy static files."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
SAMPLES = ROOT / "samples"
SAMPLES.mkdir(exist_ok=True)

src = STATIC / "Student-Feedback-Form.csv"
df = pd.read_csv(src)
for col in df.columns:
    name = str(col).lower()
    if "name" == name or name.endswith(" name"):
        df[col] = [f"Student {i}" for i in range(1, len(df) + 1)]
    elif "email" in name:
        df[col] = [f"student{i}@example.com" for i in range(1, len(df) + 1)]
    elif "timestamp" in name:
        df[col] = "2021-08-01 12:00:00"
df.to_csv(SAMPLES / "Student-Feedback-Form.csv", index=False)

form_pdf = STATIC / "form.pdf"
if form_pdf.exists():
    shutil.copy2(form_pdf, SAMPLES / "form.pdf")

remove = [
    STATIC / "student-data.xlsx",
    STATIC / "Student-Feedback-Form.csv",
    STATIC / "1627917029057_Untitled form.csv",
    STATIC / "200.gif",
    STATIC / "paste-your-content-here.gif",
    ROOT / "drawFigure.py",
    ROOT / "templates" / "index1.html",
]
for path in STATIC.glob("plot*.jpg"):
    remove.append(path)
for path in remove:
    if path.exists():
        path.unlink()
        print("removed", path)
print("wrote", SAMPLES / "Student-Feedback-Form.csv")
