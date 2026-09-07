"""
UNIVERSAL ML MODULE - PS1: AI-Powered Feedback Intelligence Dashboard
=======================================================================
Works with ANY uploaded dataset (CSV, Excel .xlsx/.xls, or TXT) regardless
of column names. Auto-detects which column is the feedback text, which is
the category, and which is the date -- so it doesn't break if tomorrow's
dataset calls them "Comments", "Review", "Customer_Message", "Dept", "Tag",
"Submitted_On", etc.

REQUIREMENTS:
  pip install vaderSentiment scikit-learn pandas openpyxl

USAGE:
  python universal_ml_module.py path/to/your_file.csv
  python universal_ml_module.py path/to/your_file.xlsx
  python universal_ml_module.py path/to/your_file.txt
"""

import sys
import re
from typing import Optional
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# =========================================================
# STEP 1: Load any file type into a DataFrame
# =========================================================
def load_any_file(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    elif lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    elif lower.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = [line.strip() for line in f if line.strip()]
        return pd.DataFrame({"raw_text": lines})
    else:
        raise ValueError(f"Unsupported file type: {path}")


# =========================================================
# STEP 2: Auto-detect the feedback TEXT column
#   Strategy 1: column name matches known keywords
#   Strategy 2: fallback -- longest average word count among text columns
# =========================================================
FEEDBACK_NAME_HINTS = [
    "feedback", "review", "comment", "message", "complaint",
    "text", "description", "remark", "response"
]

def detect_feedback_column(df: pd.DataFrame) -> str:
    # Strategy 1: name-based match
    for col in df.columns:
        col_lower = col.lower()
        if any(hint in col_lower for hint in FEEDBACK_NAME_HINTS):
            return col

    # Strategy 2: content-based fallback
    # pick the object/text column with the highest average word count
    candidate = None
    best_avg_words = 0
    for col in df.columns:
        if df[col].dtype == object:
            avg_words = df[col].dropna().astype(str).apply(lambda x: len(x.split())).mean()
            if pd.notna(avg_words) and avg_words > best_avg_words:
                best_avg_words = avg_words
                candidate = col

    if candidate is None:
        raise ValueError("Could not detect a feedback text column in this file.")
    return candidate


# =========================================================
# STEP 3: Auto-detect the CATEGORY column
#   Strategy 1: column name matches known keywords
#   Strategy 2: fallback -- low-cardinality text column (repeated values)
# =========================================================
CATEGORY_NAME_HINTS = [
    "category", "department", "dept", "tag", "module",
    "type", "class", "topic", "section"
]

def detect_category_column(df: pd.DataFrame, feedback_col: str) -> Optional[str]:
    for col in df.columns:
        col_lower = col.lower()
        if col != feedback_col and any(hint in col_lower for hint in CATEGORY_NAME_HINTS):
            return col

    # Fallback: an object column (not the feedback column) where the
    # number of unique values is small relative to row count --
    # categories repeat, feedback text almost never does.
    candidate = None
    best_ratio = 1.0
    for col in df.columns:
        if col == feedback_col or df[col].dtype != object:
            continue
        n_unique = df[col].nunique(dropna=True)
        ratio = n_unique / max(len(df), 1)
        if ratio < 0.3 and ratio < best_ratio:
            best_ratio = ratio
            candidate = col

    return candidate  # may be None if nothing qualifies


# =========================================================
# STEP 4: Auto-detect the DATE column
#   Strategy 1: column name matches known keywords
#   Strategy 2: fallback -- try parsing as dates, check success rate
# =========================================================
DATE_NAME_HINTS = ["date", "time", "day", "timestamp", "submitted", "created", "on"]

def detect_date_column(df: pd.DataFrame, feedback_col: str, category_col) -> Optional[str]:
    for col in df.columns:
        col_lower = col.lower()
        if col not in (feedback_col, category_col) and any(hint in col_lower for hint in DATE_NAME_HINTS):
            return col

    # Fallback: try parsing every remaining column as dates,
    # keep the one with the highest successful-parse rate.
    candidate = None
    best_rate = 0.5  # require at least 50% parse success to qualify
    for col in df.columns:
        if col in (feedback_col, category_col):
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        rate = parsed.notna().mean()
        if rate > best_rate:
            best_rate = rate
            candidate = col

    return candidate


# =========================================================
# STEP 5: Clean the detected feedback text
# =========================================================
def clean_feedback_series(series: pd.Series) -> pd.Series:
    series = series.astype(str).str.strip()
    series = series.apply(lambda x: re.sub(r"\s+", " ", x))
    return series


# =========================================================
# STEP 6: Sentiment analysis (Title Case to match dashboard)
# =========================================================
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        label = "Positive"
    elif score <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return label, score


def extract_keywords(text, n=3):
    words = re.findall(r"\b[a-z]{4,}\b", str(text).lower())
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    seen = list(dict.fromkeys(words))
    return ", ".join(seen[:n])


# =========================================================
# MAIN PIPELINE
# =========================================================
def process_file(path: str, output_path: str = "feedback_for_dashboard.csv"):
    print(f"Loading: {path}")
    df = load_any_file(path)
    print(f"Loaded {len(df)} rows. Columns found: {list(df.columns)}")

    feedback_col = detect_feedback_column(df)
    category_col = detect_category_column(df, feedback_col)
    date_col = detect_date_column(df, feedback_col, category_col)

    print(f"\nDetected feedback column: '{feedback_col}'")
    print(f"Detected category column: '{category_col}'" if category_col else "No category column detected -- will use 'General'")
    print(f"Detected date column: '{date_col}'" if date_col else "No date column detected -- will leave blank")

    # Drop rows with missing/empty feedback text
    df = df.dropna(subset=[feedback_col])
    df[feedback_col] = clean_feedback_series(df[feedback_col])
    df = df[df[feedback_col].str.split().str.len() >= 3]  # drop junk/too-short rows

    # Build the standardized output frame
    out = pd.DataFrame()
    out["Feedback"] = df[feedback_col]
    out["Category"] = df[category_col].astype(str).str.strip().str.title() if category_col else "General"
    if date_col:
        out["Date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        out["Date"] = ""

    # Remove exact duplicates after cleaning
    out = out.drop_duplicates(subset=["Feedback", "Category", "Date"], keep="first").reset_index(drop=True)

    # Sentiment + keywords
    results = out["Feedback"].map(get_sentiment)
    out["Sentiment"] = [r[0] for r in results]
    out["Sentiment_Score"] = [r[1] for r in results]
    out["Top_Keywords"] = out["Feedback"].apply(extract_keywords)

    out.to_csv(output_path, index=False)

    print(f"\nFinal row count: {len(out)}")
    print("\nSentiment distribution:")
    print(out["Sentiment"].value_counts())
    print("\nCategory distribution:")
    print(out["Category"].value_counts())
    print(f"\nSaved: {output_path}")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python universal_ml_module.py <path_to_file.csv|.xlsx|.txt>")
    else:
        process_file(sys.argv[1])