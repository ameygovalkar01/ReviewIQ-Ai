import re
from collections import Counter

import streamlit as st
import pandas as pd
import altair as alt

# ----------------------------------------------------
# Page Config
# ----------------------------------------------------
st.set_page_config(
    page_title="ReviewIQ AI - Executive Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# Enterprise Professional Design System (CSS)
# ----------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: #f8fafc;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
    }
    .metric-label {
        font-size: 0.815rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .metric-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 9999px;
        margin-top: 8px;
    }

    .badge-blue { background-color: #e0e7ff; color: #4338ca; }
    .badge-green { background-color: #dcfce7; color: #15803d; }
    .badge-amber { background-color: #fef3c7; color: #b45309; }
    .badge-red { background-color: #ffe4e6; color: #be123c; }
    .badge-purple { background-color: #f3e8ff; color: #7e22ce; }

    .panel-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 22px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
    }
    .panel-title {
        font-size: 1rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 2px;
    }
    .panel-subtitle {
        font-size: 0.8rem;
        color: #64748b;
        margin-bottom: 16px;
    }

    .keyword-chip {
        display: inline-block;
        background: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fecaca;
        border-radius: 9999px;
        padding: 4px 12px;
        margin: 3px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .progress-note {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.82rem;
        color: #1e3a8a;
        margin-bottom: 10px;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

TOPIC_LABELS = ["Billing", "Customer Support", "Web App", "Mobile App", "Product Quality"]
URGENCY_LABELS = ["Critical", "High", "Medium", "Low"]
SENTIMENT_LABEL_MAP = {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"}

# How many rows get classified per button click. Kept small on purpose so a
# single run stays within free-tier CPU/RAM/time limits. Click the button
# again to process the next batch -- progress is kept between clicks.
ROWS_PER_BATCH = 20
# How many texts we hand to the zero-shot pipeline in a single call. Batched
# inference is far more efficient than one call per row per label-set.
ZS_BATCH_SIZE = 8

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "to", "of", "in", "on", "for", "with", "this", "that", "it", "its", "i", "we",
    "you", "your", "my", "our", "they", "their", "at", "as", "by", "from", "not",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "so", "very", "just", "than", "then", "there", "here", "if", "when", "what",
    "which", "who", "how", "all", "any", "some", "no", "yes", "app", "product"
}


# ----------------------------------------------------
# Cached model loaders (cache_resource: models aren't
# picklable data, so cache_data would fail/re-download)
# ----------------------------------------------------
@st.cache_resource(show_spinner="Downloading/loading sentiment model (RoBERTa)...")
def load_sentiment_model():
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True,
    )


@st.cache_resource(show_spinner="Downloading/loading zero-shot model...")
def load_zero_shot_model():
    from transformers import pipeline
    # Lighter distilled model (~250MB vs ~1.6GB for bart-large-mnli) so it
    # fits free-tier RAM. Loaded once and reused across every batch/run.
    return pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")


# ----------------------------------------------------
# Classification: batched zero-shot calls instead of one call per row.
# The HF zero-shot pipeline accepts a LIST of texts and processes them as
# one batch internally, which is dramatically faster than looping row by
# row on CPU (fewer Python-level calls, better use of each forward pass).
# ----------------------------------------------------
def classify_batch(texts: list, run_sentiment: bool, run_topic: bool, run_urgency: bool,
                    sentiment_pipe, zero_shot_pipe, progress_cb=None):
    n = len(texts)
    sentiments, sentiment_scores = [None] * n, [None] * n
    topics = [None] * n
    urgencies = [None] * n

    if run_sentiment and sentiment_pipe is not None:
        results = sentiment_pipe(texts, batch_size=16, truncation=True)
        for i, r in enumerate(results):
            label = SENTIMENT_LABEL_MAP.get(r["label"].lower(), r["label"])
            sentiments[i] = label
            sentiment_scores[i] = round(float(r["score"]), 3)
    if progress_cb:
        progress_cb(0.4 if (run_topic or run_urgency) else 1.0)

    if run_topic and zero_shot_pipe is not None:
        for start in range(0, n, ZS_BATCH_SIZE):
            sub = texts[start:start + ZS_BATCH_SIZE]
            outs = zero_shot_pipe(sub, TOPIC_LABELS, multi_label=False)
            outs = outs if isinstance(outs, list) else [outs]
            for i, out in enumerate(outs):
                topics[start + i] = out["labels"][0]
    if progress_cb:
        progress_cb(0.7 if run_urgency else 1.0)

    if run_urgency and zero_shot_pipe is not None:
        for start in range(0, n, ZS_BATCH_SIZE):
            sub = texts[start:start + ZS_BATCH_SIZE]
            outs = zero_shot_pipe(sub, URGENCY_LABELS, multi_label=False)
            outs = outs if isinstance(outs, list) else [outs]
            for i, out in enumerate(outs):
                urgencies[start + i] = out["labels"][0]
    if progress_cb:
        progress_cb(1.0)

    return sentiments, sentiment_scores, topics, urgencies


def pct(part: int, whole: int) -> float:
    return round((part / whole) * 100, 1) if whole > 0 else 0.0


def guess_text_column(df: pd.DataFrame) -> str:
    candidates = ["comment", "feedback", "review", "text", "feedback summary", "message"]
    for col in df.columns:
        if col.strip().lower() in candidates:
            return col
    object_cols = [c for c in df.columns if df[c].dtype == object]
    return object_cols[0] if object_cols else df.columns[0]


def extract_top_keywords(texts, top_n=12):
    """Simple frequency-based keyword extraction, no extra ML deps."""
    counter = Counter()
    for t in texts:
        words = re.findall(r"[a-zA-Z']{3,}", str(t).lower())
        for w in words:
            if w not in STOPWORDS:
                counter[w] += 1
    return counter.most_common(top_n)


@st.cache_data(show_spinner="Parsing uploaded file...")
def load_feedback_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(pd.io.common.BytesIO(file_bytes))
    elif name.endswith(".txt"):
        lines = file_bytes.decode("utf-8", errors="replace").splitlines()
        df = pd.DataFrame({"Comment": [line.strip() for line in lines if line.strip()]})
    else:
        raise ValueError(f"Unsupported file type: {filename}")
    return df


@st.cache_data(show_spinner=False)
def get_mock_data():
    total_count = 10000
    pos_count, neu_count, neg_count = 6000, 1500, 2500
    sentiment_df = pd.DataFrame({
        "Sentiment": ["Positive", "Neutral", "Negative"],
        "Count": [pos_count, neu_count, neg_count],
    })
    topics_df = pd.DataFrame({
        "Topic": ["Checkout & Payments", "Support Response", "Search & Navigation",
                  "Account Authentication", "Pricing Transparency"],
        "Mentions": [2840, 1920, 1450, 1100, 930]
    })
    urgency_df = pd.DataFrame({
        "Urgency": ["Critical", "High", "Medium", "Low"],
        "Count": [820, 1460, 2100, 5620]
    })
    return total_count, sentiment_df, topics_df, urgency_df


@st.cache_data(show_spinner=False)
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ----------------------------------------------------
# Sidebar: Data Controls
# ----------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 16px;'>
            <div style='background: linear-gradient(135deg, #4f46e5, #06b6d4); width: 28px; height: 28px; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 14px;'>R</div>
            <span style='font-size: 1.15rem; font-weight: 700; color: #0f172a;'>ReviewIQ <span style='color: #4f46e5;'>AI</span></span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("INGESTION & SOURCES")
    uploaded_file = st.file_uploader(
        "Upload Customer Reviews",
        type=["csv", "xlsx", "xls", "txt"],
        help="Upload tabular feedback records or plain text dumps."
    )

    text_col = None
    run_sentiment_ai = run_topic_ai = run_urgency_ai = False
    run_clicked = False
    reset_clicked = False

    if uploaded_file is not None:
        try:
            _preview_df = load_feedback_file(uploaded_file.getvalue(), uploaded_file.name)
            st.sidebar.success(f"✓ Ingested {uploaded_file.name} ({len(_preview_df):,} rows)")

            st.markdown("<hr style='margin: 18px 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)
            st.caption("AI CLASSIFICATION")

            text_col = st.selectbox(
                "Text column to analyze",
                options=list(_preview_df.columns),
                index=list(_preview_df.columns).index(guess_text_column(_preview_df))
            )

            has_sentiment_col = "Sentiment" in _preview_df.columns
            run_sentiment_ai = st.checkbox("Predict Sentiment (RoBERTa)", value=not has_sentiment_col)
            run_topic_ai = st.checkbox("Predict Topic/Category (zero-shot)", value=True)
            run_urgency_ai = st.checkbox("Predict Urgency (zero-shot)", value=True)

            state_key = f"classified::{uploaded_file.name}::{uploaded_file.size}"
            already_done = 0
            if state_key in st.session_state:
                already_done = st.session_state[state_key]["next_idx"]

            total_rows = len(_preview_df)
            remaining = total_rows - already_done

            st.markdown(
                f"<div class='progress-note'>Classified <b>{already_done:,} / {total_rows:,}</b> rows so far. "
                f"Each click processes the next <b>{min(ROWS_PER_BATCH, max(remaining, 0)):,}</b> rows "
                f"— click multiple times to cover the whole file without hitting free-tier limits.</div>",
                unsafe_allow_html=True
            )

            btn_col1, btn_col2 = st.columns([3, 1])
            with btn_col1:
                label = "🚀 Classify Next Batch" if already_done > 0 else "🚀 Run AI Classification"
                run_clicked = st.button(
                    label, use_container_width=True,
                    disabled=(remaining <= 0)
                )
            with btn_col2:
                reset_clicked = st.button("↺", help="Clear all classified results and start over", use_container_width=True)

            if remaining <= 0 and total_rows > 0:
                st.sidebar.success("✓ All rows classified.")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")

    st.markdown("<hr style='margin: 18px 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)
    st.caption("FILTER ATTRIBUTES")

    date_preset = st.selectbox(
        "Time Range",
        ["Last 30 Days", "Last 90 Days", "Year to Date", "All Time"],
        index=0
    )

    department = st.multiselect(
        "Topic/Department Scope",
        options=TOPIC_LABELS,
        default=TOPIC_LABELS
    )

    sentiment_filter = st.multiselect(
        "Sentiment Filter",
        options=["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"]
    )

    min_confidence = st.slider(
        "Min. Confidence",
        min_value=0.0, max_value=1.0, value=0.0, step=0.05,
        help="Hide rows below this sentiment-model confidence score."
    )

    search_query = st.text_input("🔍 Search feedback text", placeholder="e.g. refund, crash, slow...")

# ----------------------------------------------------
# Run classification / manage session state
# (resumable: one small batch per click, appended to what's already done)
# ----------------------------------------------------
is_custom_data = uploaded_file is not None
feedback_df = None

if is_custom_data:
    raw_df = load_feedback_file(uploaded_file.getvalue(), uploaded_file.name)
    state_key = f"classified::{uploaded_file.name}::{uploaded_file.size}"

    if reset_clicked and state_key in st.session_state:
        del st.session_state[state_key]
        st.sidebar.info("Cleared previous results.")

    if state_key not in st.session_state:
        empty = raw_df.copy()
        for col in ["Sentiment", "Sentiment_Confidence", "Topic", "Urgency"]:
            empty[col] = pd.NA
        st.session_state[state_key] = {"df": empty, "next_idx": 0}

    store = st.session_state[state_key]

    if run_clicked:
        start = store["next_idx"]
        end = min(start + ROWS_PER_BATCH, len(raw_df))
        batch_df = raw_df.iloc[start:end]

        try:
            with st.spinner("Loading models (first run can take a minute)..."):
                sentiment_pipe = load_sentiment_model() if run_sentiment_ai else None
                zero_shot_pipe = load_zero_shot_model() if (run_topic_ai or run_urgency_ai) else None

            progress = st.progress(0.0, text=f"Classifying rows {start+1}-{end}...")
            texts = batch_df[text_col].astype(str).fillna("").tolist()
            sentiments, scores, topics, urgencies = classify_batch(
                texts, run_sentiment_ai, run_topic_ai, run_urgency_ai,
                sentiment_pipe, zero_shot_pipe,
                progress_cb=lambda p: progress.progress(p, text=f"Classifying rows {start+1}-{end}...")
            )
            progress.empty()

            idx = batch_df.index
            if run_sentiment_ai:
                store["df"].loc[idx, "Sentiment"] = sentiments
                store["df"].loc[idx, "Sentiment_Confidence"] = scores
            elif store["df"].loc[idx, "Sentiment"].isna().all():
                store["df"].loc[idx, "Sentiment"] = "Neutral"

            if run_topic_ai:
                store["df"].loc[idx, "Topic"] = topics
            if run_urgency_ai:
                store["df"].loc[idx, "Urgency"] = urgencies

            store["next_idx"] = end
            st.session_state[state_key] = store
            st.sidebar.success(f"✓ Classified rows {start+1}-{end} ({end:,}/{len(raw_df):,} total)")
        except ModuleNotFoundError as e:
            st.sidebar.error(
                f"Missing dependency: {e}. Add 'transformers' and 'torch' to requirements.txt "
                "and redeploy the app."
            )
        except Exception as e:
            st.sidebar.error(f"Classification failed: {type(e).__name__}: {e}")

    feedback_df_full = st.session_state[state_key]["df"]
    # Only rows that have actually been classified feed the analytics below.
    feedback_df = feedback_df_full[feedback_df_full["Sentiment"].notna()].copy()

# ----------------------------------------------------
# Build display data (apply filters)
# ----------------------------------------------------
if not is_custom_data:
    total_count, sentiment_df, topics_df, urgency_df = get_mock_data()
    if sentiment_filter and set(sentiment_filter) != {"Positive", "Neutral", "Negative"}:
        sentiment_df = sentiment_df[sentiment_df["Sentiment"].isin(sentiment_filter)]
        total_count = int(sentiment_df["Count"].sum())
    avg_confidence = None
    topic_sentiment_df = pd.DataFrame()
else:
    filtered = feedback_df.copy()
    if "Sentiment" in filtered.columns and sentiment_filter:
        filtered = filtered[filtered["Sentiment"].isin(sentiment_filter)]
    if "Topic" in filtered.columns and department:
        filtered = filtered[filtered["Topic"].isin(department) | filtered["Topic"].isna()]
    if "Sentiment_Confidence" in filtered.columns and min_confidence > 0:
        filtered = filtered[filtered["Sentiment_Confidence"].fillna(0) >= min_confidence]
    if search_query and text_col and text_col in filtered.columns:
        filtered = filtered[filtered[text_col].astype(str).str.contains(search_query, case=False, na=False)]

    total_count = len(filtered)

    if "Sentiment" in filtered.columns:
        counts = filtered["Sentiment"].value_counts()
        sentiment_df = pd.DataFrame({
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Count": [int(counts.get("Positive", 0)), int(counts.get("Neutral", 0)), int(counts.get("Negative", 0))]
        })
    else:
        sentiment_df = pd.DataFrame({"Sentiment": ["Positive", "Neutral", "Negative"], "Count": [0, 0, 0]})

    if "Topic" in filtered.columns:
        topics_df = filtered["Topic"].dropna().value_counts().reset_index()
        topics_df.columns = ["Topic", "Mentions"]
    else:
        topics_df = pd.DataFrame({"Topic": [], "Mentions": []})

    if "Urgency" in filtered.columns:
        urgency_counts = filtered["Urgency"].value_counts()
        urgency_df = pd.DataFrame({
            "Urgency": URGENCY_LABELS,
            "Count": [int(urgency_counts.get(u, 0)) for u in URGENCY_LABELS]
        })
    else:
        urgency_df = pd.DataFrame({"Urgency": [], "Count": []})

    avg_confidence = (
        round(float(filtered["Sentiment_Confidence"].mean()) * 100, 1)
        if "Sentiment_Confidence" in filtered.columns and filtered["Sentiment_Confidence"].notna().any()
        else None
    )

    if "Topic" in filtered.columns and "Sentiment" in filtered.columns:
        topic_sentiment_df = (
            filtered.dropna(subset=["Topic"]).groupby(["Topic", "Sentiment"]).size().reset_index(name="Count")
        )
    else:
        topic_sentiment_df = pd.DataFrame()

    feedback_df = filtered

pos_count = int(sentiment_df.loc[sentiment_df["Sentiment"] == "Positive", "Count"].sum())
neu_count = int(sentiment_df.loc[sentiment_df["Sentiment"] == "Neutral", "Count"].sum())
neg_count = int(sentiment_df.loc[sentiment_df["Sentiment"] == "Negative", "Count"].sum())
pos_pct, neu_pct, neg_pct = pct(pos_count, total_count), pct(neu_count, total_count), pct(neg_count, total_count)
sentiment_df["Color"] = sentiment_df["Sentiment"].map({"Positive": "#10b981", "Neutral": "#94a3b8", "Negative": "#f43f5e"})

# ----------------------------------------------------
# Main Header
# ----------------------------------------------------
header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.markdown(
        """
        <div style='margin-bottom: 24px;'>
            <h1 style='color: #0f172a; font-size: 2.1rem; font-weight: 700; margin-bottom: 4px; letter-spacing: -0.02em;'>
                ReviewIQ <span style='background: linear-gradient(135deg, #4f46e5, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>AI</span>
            </h1>
            <p style='color: #64748b; font-size: 0.95rem; margin: 0;'>Continuous intelligence & customer sentiment synthesis</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with header_col2:
    if is_custom_data and "Sentiment" in feedback_df.columns and not feedback_df.empty:
        st.download_button(
            "⬇️ Export CSV",
            data=convert_df_to_csv(feedback_df),
            file_name="classified_feedback.csv",
            mime="text/csv",
            use_container_width=True,
        )

if not is_custom_data:
    st.info("Showing sample demo data. Upload a file and click **Run AI Classification** in the sidebar to analyze your own reviews.", icon="ℹ️")
elif feedback_df.empty:
    st.warning("File loaded but not yet classified. Pick a text column and click **Run AI Classification** in the sidebar. Large files are classified in small batches — click the button repeatedly to keep going.", icon="⚠️")
elif search_query:
    st.caption(f"Showing results filtered by search: \"{search_query}\" — {total_count:,} matching rows.")

# ----------------------------------------------------
# KPI Cards
# ----------------------------------------------------
kpi_cols = st.columns(5) if avg_confidence is not None else st.columns(4)
with kpi_cols[0]:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Submissions</div>
        <div class="metric-value">{total_count:,}</div><span class="metric-badge badge-blue">100% Ingested</span></div>""",
        unsafe_allow_html=True)
with kpi_cols[1]:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Positive Sentiment</div>
        <div class="metric-value" style="color: #059669;">{pos_pct}%</div><span class="metric-badge badge-green">{pos_count:,} responses</span></div>""",
        unsafe_allow_html=True)
with kpi_cols[2]:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Neutral Sentiment</div>
        <div class="metric-value" style="color: #64748b;">{neu_pct}%</div><span class="metric-badge badge-amber">{neu_count:,} responses</span></div>""",
        unsafe_allow_html=True)
with kpi_cols[3]:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Negative Sentiment</div>
        <div class="metric-value" style="color: #e11d48;">{neg_pct}%</div><span class="metric-badge badge-red">{neg_count:,} responses</span></div>""",
        unsafe_allow_html=True)
if avg_confidence is not None:
    with kpi_cols[4]:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Avg. Model Confidence</div>
            <div class="metric-value" style="color: #7e22ce;">{avg_confidence}%</div><span class="metric-badge badge-purple">RoBERTa score</span></div>""",
            unsafe_allow_html=True)

st.write("")

# ----------------------------------------------------
# Charts: Sentiment / Topic / Urgency
# ----------------------------------------------------
chart_left, chart_mid, chart_right = st.columns([1, 1, 1])

with chart_left:
    st.markdown("""<div class="panel-box"><div class="panel-title">Sentiment Distribution</div>
        <div class="panel-subtitle">RoBERTa-predicted sentiment split</div>""", unsafe_allow_html=True)
    pie_chart = alt.Chart(sentiment_df).mark_arc(innerRadius=40, stroke="#ffffff", strokeWidth=2).encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(field="Sentiment", type="nominal",
            scale=alt.Scale(domain=["Positive", "Neutral", "Negative"], range=["#10b981", "#94a3b8", "#f43f5e"]),
            legend=alt.Legend(orient="bottom", title="", labelFont="Inter", labelFontSize=11)),
        tooltip=[alt.Tooltip("Sentiment"), alt.Tooltip("Count", format=",")]
    ).properties(height=240)
    st.altair_chart(pie_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with chart_mid:
    st.markdown("""<div class="panel-box"><div class="panel-title">Topic Breakdown</div>
        <div class="panel-subtitle">Zero-shot predicted categories</div>""", unsafe_allow_html=True)
    if topics_df.empty:
        st.caption("Run AI Classification with Topic prediction enabled to populate this chart.")
    else:
        topic_chart = alt.Chart(topics_df).mark_bar(cornerRadiusEnd=6, color="#4f46e5").encode(
            x=alt.X("Mentions:Q", title="Mentions", axis=alt.Axis(grid=True, gridColor="#f1f5f9")),
            y=alt.Y("Topic:N", sort="-x", title="", axis=alt.Axis(labelFont="Inter", labelFontSize=11)),
            tooltip=[alt.Tooltip("Topic"), alt.Tooltip("Mentions", format=",")]
        ).properties(height=240)
        st.altair_chart(topic_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with chart_right:
    st.markdown("""<div class="panel-box"><div class="panel-title">Urgency Breakdown</div>
        <div class="panel-subtitle">Zero-shot predicted priority level</div>""", unsafe_allow_html=True)
    if urgency_df.empty or urgency_df["Count"].sum() == 0:
        st.caption("Run AI Classification with Urgency prediction enabled to populate this chart.")
    else:
        urgency_chart = alt.Chart(urgency_df).mark_bar(cornerRadiusEnd=6).encode(
            x=alt.X("Urgency:N", sort=URGENCY_LABELS, title="", axis=alt.Axis(labelFont="Inter", labelFontSize=11)),
            y=alt.Y("Count:Q", title="Reviews"),
            color=alt.Color("Urgency:N", scale=alt.Scale(
                domain=URGENCY_LABELS, range=["#be123c", "#f97316", "#eab308", "#94a3b8"]), legend=None),
            tooltip=[alt.Tooltip("Urgency"), alt.Tooltip("Count", format=",")]
        ).properties(height=240)
        st.altair_chart(urgency_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------
# Topic x Sentiment cross-analysis + Confidence distribution
# ----------------------------------------------------
cross_left, cross_right = st.columns([1, 1])

with cross_left:
    st.markdown("""<div class="panel-box"><div class="panel-title">Sentiment by Topic</div>
        <div class="panel-subtitle">Where negative sentiment concentrates across categories</div>""", unsafe_allow_html=True)
    if is_custom_data and not topic_sentiment_df.empty:
        stacked_chart = alt.Chart(topic_sentiment_df).mark_bar().encode(
            x=alt.X("Count:Q", title="Reviews", stack="normalize", axis=alt.Axis(format="%")),
            y=alt.Y("Topic:N", title="", sort="-x"),
            color=alt.Color("Sentiment:N", scale=alt.Scale(
                domain=["Positive", "Neutral", "Negative"], range=["#10b981", "#94a3b8", "#f43f5e"]),
                legend=alt.Legend(orient="bottom", title="")),
            tooltip=[alt.Tooltip("Topic"), alt.Tooltip("Sentiment"), alt.Tooltip("Count", format=",")]
        ).properties(height=240)
        st.altair_chart(stacked_chart, use_container_width=True)
    else:
        st.caption("Run AI Classification with both Topic and Sentiment enabled to populate this chart.")
    st.markdown("</div>", unsafe_allow_html=True)

with cross_right:
    st.markdown("""<div class="panel-box"><div class="panel-title">Model Confidence Distribution</div>
        <div class="panel-subtitle">How confident the sentiment model was, per prediction</div>""", unsafe_allow_html=True)
    if is_custom_data and "Sentiment_Confidence" in feedback_df.columns and feedback_df["Sentiment_Confidence"].notna().any():
        conf_chart = alt.Chart(feedback_df.dropna(subset=["Sentiment_Confidence"])).mark_bar(color="#7e22ce", cornerRadiusEnd=4).encode(
            x=alt.X("Sentiment_Confidence:Q", bin=alt.Bin(maxbins=20), title="Confidence Score"),
            y=alt.Y("count():Q", title="Reviews"),
            tooltip=[alt.Tooltip("count():Q", title="Reviews")]
        ).properties(height=240)
        st.altair_chart(conf_chart, use_container_width=True)
    else:
        st.caption("Run AI Classification with Sentiment prediction enabled to populate this chart.")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------
# Priority Action Items (real, ML-derived) + Summary
# ----------------------------------------------------
row2_col1, row2_col2 = st.columns([1, 1])

with row2_col1:
    st.markdown("""<div class="panel-box"><div class="panel-title">⚠️ Priority Action Items</div>
        <div class="panel-subtitle">Highest-urgency negative reviews from the classified data</div>""", unsafe_allow_html=True)

    if is_custom_data and "Urgency" in feedback_df.columns and "Sentiment" in feedback_df.columns:
        priority = feedback_df[
            feedback_df["Urgency"].isin(["Critical", "High"]) & (feedback_df["Sentiment"] == "Negative")
        ]
        if priority.empty:
            st.caption("No critical/high-urgency negative reviews found in the classified sample so far.")
        else:
            show_cols = [c for c in [text_col, "Topic", "Urgency", "Sentiment_Confidence"] if c and c in priority.columns]
            st.dataframe(priority[show_cols].head(10), use_container_width=True, hide_index=True)

        if text_col and not priority.empty:
            keywords = extract_top_keywords(priority[text_col].tolist(), top_n=12)
            if keywords:
                st.markdown("<div style='margin-top:12px; font-size:0.8rem; color:#64748b; font-weight:600;'>TOP KEYWORDS IN NEGATIVE FEEDBACK</div>", unsafe_allow_html=True)
                chips = "".join(f"<span class='keyword-chip'>{word} ({count})</span>" for word, count in keywords)
                st.markdown(f"<div style='margin-top:6px;'>{chips}</div>", unsafe_allow_html=True)
    elif not is_custom_data:
        issues_df = pd.DataFrame([
            {"Issue": "Gateway timeout during checkout", "Severity": "P1 - Critical", "Tickets": 820, "Status": "Open"},
            {"Issue": "Session timeout during login on iOS", "Severity": "P2 - High", "Tickets": 640, "Status": "Investigating"},
            {"Issue": "Delayed chat support queues", "Severity": "P3 - Medium", "Tickets": 530, "Status": "Backlog"},
            {"Issue": "Unclear invoice charge descriptions", "Severity": "P4 - Low", "Tickets": 310, "Status": "Resolved"}
        ])
        st.dataframe(issues_df, use_container_width=True, hide_index=True)
    else:
        st.caption("Run AI Classification (Sentiment + Urgency) to populate real priority items.")
    st.markdown("</div>", unsafe_allow_html=True)

with row2_col2:
    if is_custom_data and "Sentiment" in feedback_df.columns and not feedback_df.empty:
        top_topic_line = (
            f"<p><strong>• Top Category:</strong> \"{topics_df.iloc[0]['Topic']}\" has the most mentions ({int(topics_df.iloc[0]['Mentions']):,}).</p>"
            if not topics_df.empty else ""
        )
        confidence_line = (
            f"<p><strong>• Model Confidence:</strong> Average confidence across predictions is {avg_confidence}%.</p>"
            if avg_confidence is not None else ""
        )
        critical_count = 0
        if "Urgency" in feedback_df.columns and "Sentiment" in feedback_df.columns:
            critical_count = int(
                ((feedback_df["Urgency"] == "Critical") & (feedback_df["Sentiment"] == "Negative")).sum()
            )
        critical_line = (
            f"<p><strong>• Immediate Attention:</strong> {critical_count} review(s) flagged Critical + Negative.</p>"
            if "Urgency" in feedback_df.columns else ""
        )
        html = f"""<div class="panel-box"><div class="panel-title">Data Summary</div>
            <div class="panel-subtitle">Computed directly from AI-classified data</div>
            <div style="font-size: 0.885rem; color: #334155; line-height: 1.6;">
                <p><strong>• Sentiment Split:</strong> {pos_pct}% positive, {neu_pct}% neutral, {neg_pct}% negative across {total_count:,} classified rows.</p>
                {top_topic_line}
                {confidence_line}
                {critical_line}
            </div></div>"""
    else:
        html = f"""<div class="panel-box"><div class="panel-title">Executive Synthesis</div>
            <div class="panel-subtitle">Sample narrative shown for demo data only</div>
            <div style="font-size: 0.885rem; color: #334155; line-height: 1.6;">
                <p><strong>• Sentiment Baseline:</strong> Favorable customer perception stands at <strong>{pos_pct}%</strong>.</p>
                <p><strong>• Primary Risk Vector:</strong> Negative sentiment (<strong>{neg_pct}%</strong>) is predominantly transactional.</p>
            </div></div>"""
    st.markdown(html, unsafe_allow_html=True)

# ----------------------------------------------------
# Explorer
# ----------------------------------------------------
with st.expander("Explore Classified Feedback", expanded=bool(search_query)):
    if is_custom_data and feedback_df is not None and not feedback_df.empty:
        sort_options = [c for c in feedback_df.columns if c in
                         (["Sentiment_Confidence", "Sentiment", "Topic", "Urgency"] + [text_col])]
        if sort_options:
            sort_col1, sort_col2 = st.columns([2, 1])
            with sort_col1:
                sort_by = st.selectbox("Sort by", options=sort_options, index=0, key="explorer_sort")
            with sort_col2:
                sort_dir = st.radio("Order", ["Desc", "Asc"], horizontal=True, key="explorer_sort_dir")
            display_df = feedback_df.sort_values(by=sort_by, ascending=(sort_dir == "Asc"))
        else:
            display_df = feedback_df
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    elif is_custom_data and feedback_df is not None and feedback_df.empty:
        st.caption("No classified rows yet, or none match the current filters/search.")
    else:
        st.caption("Upload and classify a file to see raw rows here.")

# ======================================================
# NEW FEATURE: Downloadable Detailed Report (Word / PDF)
# Everything below is additive -- nothing above this line
# was changed. Requires 'python-docx' and 'reportlab' to
# be present in requirements.txt.
# ======================================================
import io
from datetime import datetime


def build_report_sections() -> dict:
    """Gather everything needed for the report from data already computed above."""
    priority_rows = None
    keywords = None
    if is_custom_data and feedback_df is not None and "Urgency" in feedback_df.columns and "Sentiment" in feedback_df.columns:
        priority_rows = feedback_df[
            feedback_df["Urgency"].isin(["Critical", "High"]) & (feedback_df["Sentiment"] == "Negative")
        ]
        if text_col and not priority_rows.empty:
            keywords = extract_top_keywords(priority_rows[text_col].tolist(), top_n=12)

    return {
        "generated_at": datetime.now().strftime("%B %d, %Y %I:%M %p"),
        "total_count": total_count,
        "pos_pct": pos_pct, "neu_pct": neu_pct, "neg_pct": neg_pct,
        "pos_count": pos_count, "neu_count": neu_count, "neg_count": neg_count,
        "avg_confidence": avg_confidence,
        "sentiment_df": sentiment_df,
        "topics_df": topics_df,
        "urgency_df": urgency_df,
        "is_custom_data": is_custom_data,
        "text_col": text_col,
        "priority_rows": priority_rows,
        "keywords": keywords,
    }


def _fmt_cell(val):
    return f"{val:,}" if isinstance(val, (int, float)) and not isinstance(val, bool) else str(val)


def generate_docx_report(sections: dict) -> bytes:
    from docx import Document

    doc = Document()
    doc.add_heading("ReviewIQ AI \u2014 Detailed Analytics Report", level=0)
    gen_p = doc.add_paragraph(f"Generated on {sections['generated_at']}")
    gen_p.runs[0].italic = True

    doc.add_heading("Executive Summary", level=1)
    doc.add_paragraph(
        f"This report covers {sections['total_count']:,} classified submissions. "
        f"Sentiment breakdown: {sections['pos_pct']}% positive ({sections['pos_count']:,}), "
        f"{sections['neu_pct']}% neutral ({sections['neu_count']:,}), "
        f"{sections['neg_pct']}% negative ({sections['neg_count']:,})."
    )
    if sections["avg_confidence"] is not None:
        doc.add_paragraph(f"Average model confidence across sentiment predictions: {sections['avg_confidence']}%.")
    if not sections["is_custom_data"]:
        doc.add_paragraph("Note: this report reflects sample demo data. Upload and classify a file for a live report.")

    def add_df_table(heading, df, cols):
        doc.add_heading(heading, level=1)
        if df is None or df.empty:
            doc.add_paragraph("No data available.")
            return
        table = doc.add_table(rows=1, cols=len(cols))
        table.style = "Light Grid Accent 1"
        for i, c in enumerate(cols):
            table.rows[0].cells[i].text = str(c)
        for _, row in df.iterrows():
            cells = table.add_row().cells
            for i, c in enumerate(cols):
                cells[i].text = _fmt_cell(row[c])

    add_df_table("Sentiment Distribution", sections["sentiment_df"][["Sentiment", "Count"]], ["Sentiment", "Count"])
    add_df_table("Topic Breakdown", sections["topics_df"], ["Topic", "Mentions"])
    add_df_table("Urgency Breakdown", sections["urgency_df"], ["Urgency", "Count"])

    doc.add_heading("Priority Action Items", level=1)
    priority_rows = sections["priority_rows"]
    if priority_rows is not None and not priority_rows.empty:
        cols = [c for c in [sections["text_col"], "Topic", "Urgency", "Sentiment_Confidence"]
                if c and c in priority_rows.columns]
        table = doc.add_table(rows=1, cols=len(cols))
        table.style = "Light Grid Accent 1"
        for i, c in enumerate(cols):
            table.rows[0].cells[i].text = str(c)
        for _, row in priority_rows.head(20).iterrows():
            cells = table.add_row().cells
            for i, c in enumerate(cols):
                cells[i].text = _fmt_cell(row[c])
    else:
        doc.add_paragraph("No critical/high-urgency negative reviews found in the classified sample.")

    if sections["keywords"]:
        doc.add_heading("Top Keywords in Negative Feedback", level=1)
        doc.add_paragraph(", ".join(f"{w} ({c})" for w, c in sections["keywords"]))

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


def generate_pdf_report(sections: dict) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], textColor=colors.HexColor("#0f172a"))
    h_style = ParagraphStyle("HeadingX", parent=styles["Heading2"],
                              textColor=colors.HexColor("#4f46e5"), spaceBefore=14, spaceAfter=6)
    body_style = styles["BodyText"]

    story = [
        Paragraph("ReviewIQ AI \u2014 Detailed Analytics Report", title_style),
        Paragraph(f"Generated on {sections['generated_at']}", body_style),
        Spacer(1, 14),
        Paragraph("Executive Summary", h_style),
        Paragraph(
            f"This report covers {sections['total_count']:,} classified submissions. "
            f"Sentiment breakdown: {sections['pos_pct']}% positive ({sections['pos_count']:,}), "
            f"{sections['neu_pct']}% neutral ({sections['neu_count']:,}), "
            f"{sections['neg_pct']}% negative ({sections['neg_count']:,}).", body_style),
    ]
    if sections["avg_confidence"] is not None:
        story.append(Paragraph(f"Average model confidence across sentiment predictions: {sections['avg_confidence']}%.", body_style))
    if not sections["is_custom_data"]:
        story.append(Paragraph("Note: this report reflects sample demo data.", body_style))

    def df_table(df, cols):
        data = [cols] + [[_fmt_cell(row[c]) for c in cols] for _, row in df.iterrows()]
        t = Table(data, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        return t

    story.append(Paragraph("Sentiment Distribution", h_style))
    story.append(df_table(sections["sentiment_df"][["Sentiment", "Count"]], ["Sentiment", "Count"]))

    if not sections["topics_df"].empty:
        story.append(Paragraph("Topic Breakdown", h_style))
        story.append(df_table(sections["topics_df"], ["Topic", "Mentions"]))

    if not sections["urgency_df"].empty:
        story.append(Paragraph("Urgency Breakdown", h_style))
        story.append(df_table(sections["urgency_df"], ["Urgency", "Count"]))

    story.append(Paragraph("Priority Action Items", h_style))
    priority_rows = sections["priority_rows"]
    if priority_rows is not None and not priority_rows.empty:
        cols = [c for c in [sections["text_col"], "Topic", "Urgency", "Sentiment_Confidence"]
                if c and c in priority_rows.columns]
        rows_data = priority_rows.head(20)[cols].copy()
        for c in cols:
            rows_data[c] = rows_data[c].astype(str).str.slice(0, 60)
        story.append(df_table(rows_data, cols))
    else:
        story.append(Paragraph("No critical/high-urgency negative reviews found in the classified sample.", body_style))

    if sections["keywords"]:
        story.append(Paragraph("Top Keywords in Negative Feedback", h_style))
        story.append(Paragraph(", ".join(f"{w} ({c})" for w, c in sections["keywords"]), body_style))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


st.markdown("<hr style='margin: 24px 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)
st.markdown("""<div class="panel-box"><div class="panel-title">\U0001F4C4 Detailed Report</div>
    <div class="panel-subtitle">Export the full analytics summary as a Word or PDF document</div>""", unsafe_allow_html=True)

_report_sections = build_report_sections()
report_btn_col1, report_btn_col2 = st.columns(2)

try:
    _docx_bytes = generate_docx_report(_report_sections)
    _docx_available = True
except ModuleNotFoundError:
    _docx_bytes, _docx_available = None, False

try:
    _pdf_bytes = generate_pdf_report(_report_sections)
    _pdf_available = True
except ModuleNotFoundError:
    _pdf_bytes, _pdf_available = None, False

with report_btn_col1:
    if _docx_available:
        st.download_button(
            "\u2b07\ufe0f Download Report (Word)",
            data=_docx_bytes,
            file_name="reviewiq_detailed_report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
    else:
        st.caption("Word export needs the 'python-docx' package. Add it to requirements.txt and redeploy.")

with report_btn_col2:
    if _pdf_available:
        st.download_button(
            "\u2b07\ufe0f Download Report (PDF)",
            data=_pdf_bytes,
            file_name="reviewiq_detailed_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.caption("PDF export needs the 'reportlab' package. Add it to requirements.txt and redeploy.")

st.markdown("</div>", unsafe_allow_html=True)