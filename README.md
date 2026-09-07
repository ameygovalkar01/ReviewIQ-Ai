# ReviewIQ AI — Feedback Intelligence Dashboard

AI-powered dashboard that ingests customer feedback (CSV, Excel, or TXT) and
classifies it by **sentiment**, **topic**, and **urgency**, then visualizes
the results for stakeholders.

## Project Structure

```
ReviewIQ-AI/
├── dashboard.py              # Streamlit frontend (entry point)
├── MLModel.py                # ML classification module (VADER + sklearn)
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Theme + server settings
├── sample_reviews.csv        # Demo dataset (optional but recommended)
└── README.md
```

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public, or private with Streamlit Cloud access
   granted).
2. Make sure `venv/` is **not** committed — confirm it's in `.gitignore`.
3. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
4. Select the repo, branch, and set **Main file path** to `dashboard.py`.
5. If the app uses any API keys or secrets, add them under
   **App settings → Secrets** (do not commit a `secrets.toml` file to
   GitHub — Streamlit Cloud injects secrets the same way it would read
   from `.streamlit/secrets.toml` locally).
6. Deploy. First build installs everything in `requirements.txt`.

## Notes

- No system-level packages are required for this stack (`vaderSentiment`,
  `scikit-learn`, `pandas` are all pure-Python-installable), so no
  `packages.txt` is needed.
- If you add any API-based model later (e.g. calling an LLM), you will
  need `.streamlit/secrets.toml` locally (gitignored) and the matching
  entry in the Cloud app's Secrets manager.
