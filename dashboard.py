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
CHUNK_SIZE = 16  # batch size for progress updates


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
    # Lighter distilled model (~250MB vs ~1.6GB for bart-large-mnli)
    # so it fits in Streamlit Community Cloud's free-tier RAM budget.
    return pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")


# ----------------------------------------------------
# Classification pipeline
# ----------------------------------------------------
# Leading-underscore args (_sentiment_pipe, _zero_shot_pipe) are NOT
# hashed by st.cache_data -- only `texts`, `run_*` flags are, so this
# only recomputes when the actual input text / options change.
@st.cache_data(show_spinner=False)
def classify_texts(texts: tuple, run_sentiment: bool, run_topic: bool, run_urgency: bool,
                    _sentiment_pipe, _zero_shot_pipe):
    n = len(texts)
    sentiments, sentiment_scores = [None] * n, [None] * n
    topics = [None] * n
    urgencies = [None] * n

    progress = st.progress(0.0, text="Running AI classification...")
    total_steps = n
    done = 0

    for start in range(0, n, CHUNK_SIZE):
        chunk = list(texts[start:start + CHUNK_SIZE])

        if run_sentiment:
            results = _sentiment_pipe(chunk, batch_size=CHUNK_SIZE, truncation=True)
            for i, r in enumerate(results):
                label = SENTIMENT_LABEL_MAP.get(r["label"].lower(), r["label"])
                sentiments[start + i] = label
                sentiment_scores[start + i] = round(float(r["score"]), 3)

        if run_topic:
            for i, text in enumerate(chunk):
                out = _zero_shot_pipe(text, TOPIC_LABELS, multi_label=False)
                topics[start + i] = out["labels"][0]

        if run_urgency:
            for i, text in enumerate(chunk):
                out = _zero_shot_pipe(text, URGENCY_LABELS, multi_label=False)
                urgencies[start + i] = out["labels"][0]

        done += len(chunk)
        progress.progress(min(done / total_steps, 1.0), text=f"Classified {done}/{total_steps} rows...")

    progress.empty()
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
    max_rows = 200
    run_clicked = False

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

            max_rows = st.slider(
                "Max rows to classify (speed control)",
                min_value=20, max_value=min(2000, max(20, len(_preview_df))),
                value=min(200, len(_preview_df)), step=20,
                help="Zero-shot classification is slow on CPU. Lower this for a quick preview; raise it for full coverage."
            )

            run_clicked = st.button("🚀 Run AI Classification", use_container_width=True)
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

# ----------------------------------------------------
# Run classification / manage session state
# ----------------------------------------------------
is_custom_data = uploaded_file is not None
feedback_df = None

if is_custom_data:
    raw_df = load_feedback_file(uploaded_file.getvalue(), uploaded_file.name)
    state_key = f"classified::{uploaded_file.name}::{uploaded_file.size}"

    if run_clicked:
        sample_df = raw_df.head(max_rows).copy()

        try:
            sentiment_pipe = load_sentiment_model() if run_sentiment_ai else None
            zero_shot_pipe = load_zero_shot_model() if (run_topic_ai or run_urgency_ai) else None

            texts = tuple(sample_df[text_col].astype(str).fillna("").tolist())
            sentiments, scores, topics, urgencies = classify_texts(
                texts, run_sentiment_ai, run_topic_ai, run_urgency_ai,
                sentiment_pipe, zero_shot_pipe
            )

            if run_sentiment_ai:
                sample_df["Sentiment"] = sentiments
                sample_df["Sentiment_Confidence"] = scores
            elif "Sentiment" not in sample_df.columns:
                sample_df["Sentiment"] = "Neutral"

            if run_topic_ai:
                sample_df["Topic"] = topics
            if run_urgency_ai:
                sample_df["Urgency"] = urgencies

            st.session_state[state_key] = sample_df
            st.sidebar.success(f"✓ Classified {len(sample_df):,} rows")
        except ModuleNotFoundError as e:
            st.sidebar.error(
                f"Missing dependency: {e}. Add 'transformers' and 'torch' to requirements.txt "
                "and redeploy the app."
            )
        except Exception as e:
            st.sidebar.error(f"Classification failed: {e}")

    feedback_df = st.session_state.get(state_key, raw_df)

# ----------------------------------------------------
# Build display data (apply filters)
# ----------------------------------------------------
if not is_custom_data:
    total_count, sentiment_df, topics_df, urgency_df = get_mock_data()
    if sentiment_filter and set(sentiment_filter) != {"Positive", "Neutral", "Negative"}:
        sentiment_df = sentiment_df[sentiment_df["Sentiment"].isin(sentiment_filter)]
        total_count = int(sentiment_df["Count"].sum())
else:
    filtered = feedback_df.copy()
    if "Sentiment" in filtered.columns and sentiment_filter:
        filtered = filtered[filtered["Sentiment"].isin(sentiment_filter)]
    if "Topic" in filtered.columns and department:
        filtered = filtered[filtered["Topic"].isin(department)]

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
        topics_df = filtered["Topic"].value_counts().reset_index()
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

    feedback_df = filtered

pos_count = int(sentiment_df.loc[sentiment_df["Sentiment"] == "Positive", "Count"].sum())
neu_count = int(sentiment_df.loc[sentiment_df["Sentiment"] == "Neutral", "Count"].sum())
neg_count = int(sentiment_df.loc[sentiment_df["Sentiment"] == "Negative", "Count"].sum())
pos_pct, neu_pct, neg_pct = pct(pos_count, total_count), pct(neu_count, total_count), pct(neg_count, total_count)
sentiment_df["Color"] = sentiment_df["Sentiment"].map({"Positive": "#10b981", "Neutral": "#94a3b8", "Negative": "#f43f5e"})

# ----------------------------------------------------
# Main Header
# ----------------------------------------------------
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

if not is_custom_data:
    st.info("Showing sample demo data. Upload a file and click **Run AI Classification** in the sidebar to analyze your own reviews.", icon="ℹ️")
elif "Sentiment" not in feedback_df.columns and "Topic" not in feedback_df.columns:
    st.warning("File loaded but not yet classified. Pick a text column and click **Run AI Classification** in the sidebar.", icon="⚠️")

# ----------------------------------------------------
# KPI Cards
# ----------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Submissions</div>
        <div class="metric-value">{total_count:,}</div><span class="metric-badge badge-blue">100% Ingested</span></div>""",
        unsafe_allow_html=True)
with kpi2:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Positive Sentiment</div>
        <div class="metric-value" style="color: #059669;">{pos_pct}%</div><span class="metric-badge badge-green">{pos_count:,} responses</span></div>""",
        unsafe_allow_html=True)
with kpi3:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Neutral Sentiment</div>
        <div class="metric-value" style="color: #64748b;">{neu_pct}%</div><span class="metric-badge badge-amber">{neu_count:,} responses</span></div>""",
        unsafe_allow_html=True)
with kpi4:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Negative Sentiment</div>
        <div class="metric-value" style="color: #e11d48;">{neg_pct}%</div><span class="metric-badge badge-red">{neg_count:,} responses</span></div>""",
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
            st.caption("No critical/high-urgency negative reviews found in the classified sample.")
        else:
            show_cols = [c for c in [text_col, "Topic", "Urgency", "Sentiment_Confidence"] if c and c in priority.columns]
            st.dataframe(priority[show_cols].head(10), use_container_width=True, hide_index=True)
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
    if is_custom_data and "Sentiment" in feedback_df.columns:
        top_topic_line = (
            f"<p><strong>• Top Category:</strong> \"{topics_df.iloc[0]['Topic']}\" has the most mentions ({int(topics_df.iloc[0]['Mentions']):,}).</p>"
            if not topics_df.empty else ""
        )
        html = f"""<div class="panel-box"><div class="panel-title">Data Summary</div>
            <div class="panel-subtitle">Computed directly from AI-classified data</div>
            <div style="font-size: 0.885rem; color: #334155; line-height: 1.6;">
                <p><strong>• Sentiment Split:</strong> {pos_pct}% positive, {neu_pct}% neutral, {neg_pct}% negative across {total_count:,} rows.</p>
                {top_topic_line}
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
with st.expander("Explore Classified Feedback"):
    if is_custom_data and feedback_df is not None:
        st.dataframe(feedback_df, use_container_width=True, hide_index=True)
    else:
        st.caption("Upload and classify a file to see raw rows here.")