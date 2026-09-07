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


# ----------------------------------------------------
# Cached data-loading / processing functions
# ----------------------------------------------------
# Caching by raw bytes + filename means Streamlit only re-parses the
# uploaded file when its content actually changes, not on every
# widget interaction / rerun.
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
        df["Sentiment"] = "Neutral"
        df["Module"] = "General"
    else:
        raise ValueError(f"Unsupported file type: {filename}")
    return df


@st.cache_data(show_spinner=False)
def get_mock_data():
    """Static demo dataset shown until a real file is uploaded."""
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

    return total_count, sentiment_df, topics_df


@st.cache_data(show_spinner=False)
def summarize_custom_data(df: pd.DataFrame, sentiment_filter: tuple, department: tuple):
    """Apply the sidebar filters to real uploaded data and compute counts."""
    filtered = df.copy()

    if "Sentiment" in filtered.columns and sentiment_filter:
        filtered = filtered[filtered["Sentiment"].isin(sentiment_filter)]

    # Only filter on department/module if the uploaded data actually has
    # a matching column -- the demo option list won't exist in most
    # real uploads, so this is applied opportunistically.
    for col in ("Department", "Module"):
        if col in filtered.columns and department:
            filtered = filtered[filtered[col].isin(department)]
            break

    total_count = len(filtered)

    if "Sentiment" in filtered.columns and total_count > 0:
        counts = filtered["Sentiment"].value_counts()
        pos_count = int(counts.get("Positive", 0))
        neu_count = int(counts.get("Neutral", 0))
        neg_count = int(counts.get("Negative", 0))
    else:
        # No sentiment column to work with -- don't fabricate a 60/15/25
        # split, just report everything as unclassified.
        pos_count = neu_count = neg_count = 0

    sentiment_df = pd.DataFrame({
        "Sentiment": ["Positive", "Neutral", "Negative"],
        "Count": [pos_count, neu_count, neg_count],
    })

    # Real topic modelling is out of scope here -- rather than inventing
    # labelled categories that don't correspond to anything in the data,
    # surface whatever categorical column looks most relevant if present.
    topic_col = next((c for c in ("Category", "Topic", "Module") if c in filtered.columns), None)
    if topic_col:
        topics_df = (
            filtered[topic_col].value_counts().head(6).reset_index()
        )
        topics_df.columns = ["Topic", "Mentions"]
    else:
        topics_df = pd.DataFrame({"Topic": [], "Mentions": []})

    return filtered, total_count, sentiment_df, topics_df


def pct(part: int, whole: int) -> float:
    return round((part / whole) * 100, 1) if whole > 0 else 0.0


# ----------------------------------------------------
# Sidebar: Professional Data Controls
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

    st.markdown("<hr style='margin: 18px 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)
    st.caption("FILTER ATTRIBUTES")

    date_preset = st.selectbox(
        "Time Range",
        ["Last 30 Days", "Last 90 Days", "Year to Date", "All Time"],
        index=0,
        help="Applied only if your file has a date column matching this range (not yet wired up)."
    )

    department = st.multiselect(
        "Department Scope",
        options=["Billing", "Customer Support", "Web App", "Mobile (iOS/Android)", "Product Quality"],
        default=["Billing", "Customer Support", "Web App", "Mobile (iOS/Android)", "Product Quality"]
    )

    sentiment_filter = st.multiselect(
        "Sentiment Filter",
        options=["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"]
    )

# ----------------------------------------------------
# Data Parsing & State Logic
# ----------------------------------------------------
is_custom_data = False
feedback_df = None

if uploaded_file is not None:
    try:
        feedback_df = load_feedback_file(uploaded_file.getvalue(), uploaded_file.name)
        is_custom_data = True
        st.sidebar.success(f"✓ Ingested {uploaded_file.name}")
    except ValueError as e:
        st.sidebar.error(str(e))
    except ImportError:
        st.sidebar.error("Reading .xlsx/.xls requires the 'openpyxl' package. Try a CSV instead, or install openpyxl.")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

if not is_custom_data or feedback_df is None:
    total_count, sentiment_df, topics_df = get_mock_data()
    if sentiment_filter and set(sentiment_filter) != {"Positive", "Neutral", "Negative"}:
        sentiment_df = sentiment_df[sentiment_df["Sentiment"].isin(sentiment_filter)]
        total_count = int(sentiment_df["Count"].sum())
else:
    feedback_df, total_count, sentiment_df, topics_df = summarize_custom_data(
        feedback_df, tuple(sentiment_filter), tuple(department)
    )

pos_count = int(sentiment_df.loc[sentiment_df["Sentiment"] == "Positive", "Count"].sum())
neu_count = int(sentiment_df.loc[sentiment_df["Sentiment"] == "Neutral", "Count"].sum())
neg_count = int(sentiment_df.loc[sentiment_df["Sentiment"] == "Negative", "Count"].sum())
pos_pct, neu_pct, neg_pct = pct(pos_count, total_count), pct(neu_count, total_count), pct(neg_count, total_count)

sentiment_df["Color"] = sentiment_df["Sentiment"].map({
    "Positive": "#10b981", "Neutral": "#94a3b8", "Negative": "#f43f5e"
})

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
    st.info("Showing sample demo data. Upload a file in the sidebar to analyze your own reviews.", icon="ℹ️")

# ----------------------------------------------------
# High-End Metric Cards
# ----------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Submissions</div>
        <div class="metric-value">{total_count:,}</div>
        <span class="metric-badge badge-blue">100% Ingested</span>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Positive Sentiment</div>
        <div class="metric-value" style="color: #059669;">{pos_pct}%</div>
        <span class="metric-badge badge-green">{pos_count:,} responses</span>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Neutral Sentiment</div>
        <div class="metric-value" style="color: #64748b;">{neu_pct}%</div>
        <span class="metric-badge badge-amber">{neu_count:,} responses</span>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Negative Sentiment</div>
        <div class="metric-value" style="color: #e11d48;">{neg_pct}%</div>
        <span class="metric-badge badge-red">{neg_count:,} responses</span>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ----------------------------------------------------
# Visual Analytics (Pie & Horizontal Bar Chart)
# ----------------------------------------------------
chart_left, chart_right = st.columns([1, 1])

with chart_left:
    st.markdown("""
    <div class="panel-box">
        <div class="panel-title">Sentiment Distribution</div>
        <div class="panel-subtitle">Proportional split across verified user submissions</div>
    """, unsafe_allow_html=True)

    pie_chart = alt.Chart(sentiment_df).mark_arc(innerRadius=45, stroke="#ffffff", strokeWidth=2).encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(
            field="Sentiment",
            type="nominal",
            scale=alt.Scale(
                domain=["Positive", "Neutral", "Negative"],
                range=["#10b981", "#94a3b8", "#f43f5e"]
            ),
            legend=alt.Legend(orient="bottom", title="", labelFont="Inter", labelFontSize=12)
        ),
        tooltip=[alt.Tooltip("Sentiment"), alt.Tooltip("Count", format=",")]
    ).properties(height=260)

    st.altair_chart(pie_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with chart_right:
    st.markdown("""
    <div class="panel-box">
        <div class="panel-title">Top Mentioned Categories</div>
        <div class="panel-subtitle">Thematic tags extracted across customer feedback</div>
    """, unsafe_allow_html=True)

    if topics_df.empty:
        st.caption("No categorical column (e.g. 'Category', 'Topic', 'Module') found in the uploaded data to summarize.")
    else:
        topic_chart = alt.Chart(topics_df).mark_bar(
            cornerRadiusEnd=6,
            color="#4f46e5"
        ).encode(
            x=alt.X("Mentions:Q", title="Volume of Mentions", axis=alt.Axis(grid=True, gridColor="#f1f5f9")),
            y=alt.Y("Topic:N", sort="-x", title="", axis=alt.Axis(labelFont="Inter", labelFontSize=12)),
            tooltip=[alt.Tooltip("Topic"), alt.Tooltip("Mentions", format=",")]
        ).properties(height=260)

        st.altair_chart(topic_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------
# Bottom Section: Priority Action Items & Synthesis
# ----------------------------------------------------
row2_col1, row2_col2 = st.columns([1, 1])

with row2_col1:
    st.markdown("""
    <div class="panel-box">
        <div class="panel-title">⚠️ Priority Action Items</div>
        <div class="panel-subtitle">Anomalous spikes in negative sentiment requiring triage</div>
    """, unsafe_allow_html=True)

    if is_custom_data:
        st.caption("Priority triage requires a ticketing/issue feed -- not derivable from this file alone.")
    else:
        issues_df = pd.DataFrame([
            {"Issue": "Gateway timeout during checkout", "Severity": "P1 - Critical", "Tickets": 820, "Status": "Open"},
            {"Issue": "Session timeout during login on iOS", "Severity": "P2 - High", "Tickets": 640, "Status": "Investigating"},
            {"Issue": "Delayed chat support queues", "Severity": "P3 - Medium", "Tickets": 530, "Status": "Backlog"},
            {"Issue": "Unclear invoice charge descriptions", "Severity": "P4 - Low", "Tickets": 310, "Status": "Resolved"}
        ])
        st.dataframe(
            issues_df,
            column_config={
                "Severity": st.column_config.TextColumn("Severity", width="medium"),
                "Tickets": st.column_config.NumberColumn("Volume", format="%d"),
                "Status": st.column_config.TextColumn("Status", width="small"),
            },
            use_container_width=True,
            hide_index=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

with row2_col2:
    if is_custom_data:
        top_topic_line = (
            f"<p><strong>• Top Category:</strong> \"{topics_df.iloc[0]['Topic']}\" accounts for the most mentions "
            f"({int(topics_df.iloc[0]['Mentions']):,}) in the filtered data.</p>"
            if not topics_df.empty else ""
        )
        synthesis_html = f"""
        <div class="panel-box">
            <div class="panel-title">Data Summary</div>
            <div class="panel-subtitle">Computed directly from your uploaded file (not an AI-generated narrative)</div>
            <div style="font-size: 0.885rem; color: #334155; line-height: 1.6;">
                <p><strong>• Sentiment Split:</strong> {pos_pct}% positive, {neu_pct}% neutral, {neg_pct}% negative across {total_count:,} filtered records.</p>
                {top_topic_line}
            </div>
        </div>
        """
    else:
        synthesis_html = f"""
        <div class="panel-box">
            <div class="panel-title">Executive Synthesis</div>
            <div class="panel-subtitle">Sample narrative shown for demo data only</div>
            <div style="font-size: 0.885rem; color: #334155; line-height: 1.6;">
                <p><strong>• Sentiment Baseline:</strong> Favorable customer perception stands at <strong>{pos_pct}%</strong>, anchored largely by search enhancements and responsive support personnel.</p>
                <p><strong>• Primary Risk Vector:</strong> Negative sentiment (<strong>{neg_pct}%</strong>) is predominantly transactional. Over 30% of critical complaints focus specifically on payment checkout dropped sessions.</p>
                <p style="margin-bottom: 0;"><strong>• Immediate Recommendation:</strong> Engineering should conduct a latency and timeout review on payment webhook callbacks before the next sprint release.</p>
            </div>
        </div>
        """
    st.markdown(synthesis_html, unsafe_allow_html=True)

# ----------------------------------------------------
# Expandable Data Explorer
# ----------------------------------------------------
with st.expander("Explore Raw Feedback Logs"):
    if is_custom_data and feedback_df is not None:
        st.dataframe(feedback_df, use_container_width=True, hide_index=True)
    else:
        sample_logs = pd.DataFrame({
            "Reference": ["REV-9042", "REV-9041", "REV-9040", "REV-9039"],
            "Timestamp": ["2026-09-07 10:14", "2026-09-07 09:30", "2026-09-06 18:45", "2026-09-06 14:12"],
            "Category": ["Web App", "Billing", "Mobile App", "Customer Support"],
            "Sentiment": ["Negative", "Negative", "Positive", "Positive"],
            "Feedback Summary": [
                "Gateway errors out intermittently on card 3D-secure verification step.",
                "Charged renewal rate without prior email discount reminder.",
                "Search query speed is noticeably quicker than the previous patch.",
                "Representative resolved my access ticket in under two minutes."
            ]
        })
        st.dataframe(sample_logs, use_container_width=True, hide_index=True)