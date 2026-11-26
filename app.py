import math
import streamlit as st
import joblib

from preprocessing import preprocess_multilingual

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Fake News Shield — Verification Desk",
    page_icon="📰",
    layout="wide",
)

CONFIDENCE_THRESHOLD = 0.62
MIN_MATCHED_FEATURES = 2

TRUSTED_SOURCES = [
    ("Reuters", "https://www.reuters.com"),
    ("Associated Press (AP)", "https://apnews.com"),
    ("BBC News", "https://www.bbc.com/news"),
    ("PIB Fact Check", "https://factcheck.pib.gov.in"),
    ("Alt News", "https://www.altnews.in"),
]

SAMPLES = {
    "Reuters (Real)": (
        "WASHINGTON (Reuters) - The U.S. Federal Reserve held interest rates steady on Wednesday "
        "while signaling potential adjustments later this year based on inflation and labor market "
        "data. Policymakers noted that economic activity has continued to expand at a solid pace."
    ),
    "Viral Hoax (Fake)": (
        "BREAKING SHOCKING: Secret alien base discovered under the Eiffel Tower and world leaders "
        "have been hiding the truth from citizens for decades! Share this immediately before the "
        "mainstream media deletes it!"
    ),
    "Hindi (Multilingual)": (
        "नई दिल्ली: भारतीय अंतरिक्ष अनुसंधान संगठन (इसरो) ने आज एक नए संचार उपग्रह के सफल "
        "प्रक्षेपण की घोषणा की। वैज्ञानिकों के अनुसार यह मिशन देश में डिजिटल कनेक्टिविटी को मजबूत करेगा।"
    ),
}

# ==========================================
# 2. EDITORIAL UI STYLING (HTML/CSS)
# ==========================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&family=Newsreader:opsz,wght@6..72,600;6..72,700&display=swap');

:root {
    --bg-paper: #f6f4ef;
    --bg-card: #ffffff;
    --ink: #141413;
    --ink-muted: #57534e;
    --ink-faint: #78716c;
    --border: #d6d3cb;
}

.stApp {
    background-color: var(--bg-paper);
    color: var(--ink);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.block-container {
    max-width: 1120px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
}

.masthead {
    border-bottom: 2px solid var(--ink);
    padding-bottom: 14px;
    margin-bottom: 22px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    flex-wrap: wrap;
    gap: 12px;
}

.masthead-kicker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--ink-muted);
    margin-bottom: 4px;
}

.masthead-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: var(--ink);
    line-height: 1.1;
    margin: 0;
}

.masthead-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--ink-muted);
}

.section-heading {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 10px;
}

.stTextArea textarea {
    background-color: #ffffff !important;
    color: var(--ink) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14.5px !important;
    line-height: 1.6 !important;
}

.stTextArea textarea:focus {
    border-color: var(--ink) !important;
    box-shadow: none !important;
}

.stButton > button {
    border-radius: 0px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    border: 1px solid var(--border) !important;
    background-color: #ffffff !important;
    color: var(--ink) !important;
    transition: border-color 0.15s ease;
}

.stButton > button:hover {
    border-color: var(--ink) !important;
    color: var(--ink) !important;
}

.stButton > button[kind="primary"] {
    background-color: var(--ink) !important;
    border-color: var(--ink) !important;
    color: #ffffff !important;
}

.stButton > button[kind="primary"]:hover {
    background-color: #292524 !important;
    color: #ffffff !important;
}

.verdict-box {
    padding: 18px;
    border: 1px solid var(--border);
    background: var(--bg-card);
    margin-bottom: 16px;
}

.verdict-low {
    background: #f0fdf4;
    border-color: #bbf7d0;
}

.verdict-high {
    background: #fef2f2;
    border-color: #fecaca;
}

.verdict-uncertain {
    background: #fffbeb;
    border-color: #fde68a;
}

.verdict-kicker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink-muted);
}

.verdict-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.65rem;
    font-weight: 700;
    margin: 4px 0 8px;
}

.verdict-low .verdict-title { color: #166534; }
.verdict-high .verdict-title { color: #991b1b; }
.verdict-uncertain .verdict-title { color: #92400e; }

.meter-header {
    display: flex;
    justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    color: var(--ink-muted);
    margin: 12px 0 5px;
}

.meter-track {
    height: 6px;
    background: rgba(20, 20, 19, 0.1);
    width: 100%;
}

.meter-fill {
    height: 100%;
}

.data-table {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0 18px;
    font-size: 13px;
    background: #ffffff;
    border: 1px solid var(--border);
}

.data-table th, .data-table td {
    padding: 9px 12px;
    border-bottom: 1px solid var(--border);
}

.data-table th {
    text-align: left;
    font-weight: 500;
    color: var(--ink-muted);
}

.data-table td {
    text-align: right;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--ink);
}

.sources-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px 16px;
    margin-top: 6px;
    margin-bottom: 14px;
}

.source-link {
    font-size: 13px;
    color: var(--ink) !important;
    text-decoration: underline;
    text-underline-offset: 3px;
}

.disclaimer {
    border-top: 1px solid var(--border);
    padding-top: 12px;
    font-size: 12px;
    color: var(--ink-faint);
    line-height: 1.5;
}

.trace-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    border-top: 1px solid var(--border);
    padding-top: 14px;
    margin-top: 16px;
}

.trace-item {
    font-size: 12px;
    color: var(--ink-faint);
}

.trace-item.active {
    color: var(--ink);
}

.trace-step {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    display: block;
    margin-bottom: 2px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 3. LOAD TRAINED MODEL AND VECTORIZER
# ==========================================
@st.cache_resource
def load_models():
    model = joblib.load("linear_svm_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf


try:
    model, tfidf = load_models()
except Exception as e:
    st.error(f"Error loading ML models: {e}")
    st.info("Make sure both .pkl files are present in your project folder.")
    st.stop()


# ==========================================
# 4. MASTHEAD
# ==========================================
vocab_size = len(getattr(tfidf, "vocabulary_", {}))
st.markdown(
    f"""
<div class="masthead">
    <div>
        <div class="masthead-kicker">AIP303 · Hallucination Handlers</div>
        <h1 class="masthead-title">Fake News Shield</h1>
    </div>
    <div class="masthead-meta">
        spaCy + TF-IDF ({vocab_size:,} features) · LinearSVC
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 5. WORKSPACE LAYOUT
# ==========================================
if "news_input" not in st.session_state:
    st.session_state["news_input"] = ""

col_left, col_right = st.columns([1.2, 0.85], gap="large")

with col_left:
    st.markdown('<div class="section-heading">Article Input</div>', unsafe_allow_html=True)

    sample_cols = st.columns(len(SAMPLES))
    for idx, (sample_name, sample_text) in enumerate(SAMPLES.items()):
        if sample_cols[idx].button(sample_name, key=f"sample_{idx}", use_container_width=True):
            st.session_state["news_input"] = sample_text
            st.rerun()

    news_text = st.text_area(
        "Article Text (English or Hindi)",
        key="news_input",
        placeholder="Paste the headline and body of the news article...",
        height=260,
    )

    btn_col1, btn_col2 = st.columns([1, 2])
    clear_clicked = btn_col1.button("Clear", use_container_width=True)
    predict_clicked = btn_col2.button("Run Classification", type="primary", use_container_width=True)

    if clear_clicked:
        st.session_state["news_input"] = ""
        st.session_state.pop("last_result", None)
        st.rerun()

    if predict_clicked:
        if not news_text.strip():
            st.warning("Enter or paste a news article first.")
        else:
            prep = preprocess_multilingual(news_text)
            cleaned_display = prep["cleaned_text"]
            model_input = prep["model_input_text"]

            if not cleaned_display:
                st.warning("No meaningful text found after spaCy preprocessing.")
            else:
                transformed = tfidf.transform([model_input]) if model_input else None
                matched_features = int(transformed.nnz) if transformed is not None else 0

                if transformed is None or matched_features < MIN_MATCHED_FEATURES:
                    label = "Low Confidence"
                    risk_level = "uncertain"
                    confidence = 0.50
                    raw_score = None
                    if prep["language"] != "en":
                        recommendation = (
                            f"Detected {prep['language_name']} text. The deployed TF-IDF + Linear SVM "
                            "model was trained primarily on English news corpora; insufficient English "
                            "vocabulary overlap was found. Recommend manual verification with trusted sources."
                        )
                    else:
                        recommendation = (
                            "Too few recognizable news terms were matched in the TF-IDF vocabulary. "
                            "Provide a longer article excerpt or cross-check with trusted sources."
                        )
                else:
                    prediction_code = int(model.predict(transformed)[0])
                    if hasattr(model, "decision_function"):
                        raw_score = round(float(model.decision_function(transformed)[0]), 4)
                        prob_real = 1.0 / (1.0 + math.exp(-raw_score))
                        confidence = round(max(prob_real, 1.0 - prob_real), 4)
                    else:
                        raw_score = 1.0 if prediction_code == 1 else -1.0
                        confidence = 0.75

                    if confidence < CONFIDENCE_THRESHOLD or (prep["language"] != "en" and matched_features < 5):
                        label = "Low Confidence"
                        risk_level = "uncertain"
                        recommendation = (
                            "Prediction confidence is below the reliability threshold. "
                            "Recommend user review and verification against trusted news sources."
                        )
                    elif prediction_code == 1:
                        label = "Likely Real"
                        risk_level = "low"
                        recommendation = (
                            "Lower misinformation risk detected. Still verify important claims "
                            "with independent trusted sources."
                        )
                    else:
                        label = "Likely Fake"
                        risk_level = "high"
                        recommendation = (
                            "Warning: Linguistic and TF-IDF patterns resemble misinformation in training "
                            "data. Cross-check this story with trusted fact-checking organizations."
                        )

                st.session_state["last_result"] = {
                    "label": label,
                    "risk_level": risk_level,
                    "confidence": confidence,
                    "raw_score": raw_score,
                    "language": prep["language"],
                    "language_name": prep["language_name"],
                    "token_count": prep["token_count"],
                    "matched_features": matched_features,
                    "recommendation": recommendation,
                }

    res = st.session_state.get("last_result")
    has_text = bool(news_text.strip())
    pct = int(round(res["confidence"] * 100)) if res else 0
    step2_sub = f"{res['language_name']} cleaned" if res else "Detect & lemmatize"
    step4_sub = f"{pct}% confidence" if res else "Confidence check"

    st.markdown(
        f"""
<div class="trace-grid">
    <div class="trace-item {'active' if has_text else ''}">
        <span class="trace-step">01. INPUT</span>Raw article text
    </div>
    <div class="trace-item {'active' if res else ''}">
        <span class="trace-step">02. SPACY NLP</span>{step2_sub}
    </div>
    <div class="trace-item {'active' if res else ''}">
        <span class="trace-step">03. LINEAR SVM</span>TF-IDF (1–2 grams)
    </div>
    <div class="trace-item {'active' if res else ''}">
        <span class="trace-step">04. THRESHOLD</span>{step4_sub}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

with col_right:
    st.markdown('<div class="section-heading">Assessment</div>', unsafe_allow_html=True)
    res = st.session_state.get("last_result")

    if not res:
        st.markdown(
            '<p style="color:#78716c; font-size:14px;">Submit an article on the left to inspect its '
            "language detection, TF-IDF feature overlap, and Linear SVM classification.</p>",
            unsafe_allow_html=True,
        )
    else:
        pct = int(round(res["confidence"] * 100))
        bar_color = (
            "#166534"
            if res["risk_level"] == "low"
            else "#991b1b"
            if res["risk_level"] == "high"
            else "#92400e"
        )
        raw_display = res["raw_score"] if res["raw_score"] is not None else "N/A"
        sources_html = "".join(
            f'<a class="source-link" href="{url}" target="_blank">{name}</a>'
            for name, url in TRUSTED_SOURCES
        )

        st.markdown(
            f"""
<div class="verdict-box verdict-{res['risk_level']}">
    <div class="verdict-kicker">Classifier Verdict</div>
    <div class="verdict-title">{res['label']}</div>
    <p style="font-size:13.5px; color:#141413; margin:0;">{res['recommendation']}</p>
    <div class="meter-header">
        <span>Confidence</span>
        <span>{pct}%</span>
    </div>
    <div class="meter-track">
        <div class="meter-fill" style="width:{pct}%; background-color:{bar_color};"></div>
    </div>
</div>

<table class="data-table">
    <tbody>
        <tr><th>Detected Language</th><td>{res['language_name']} ({res['language']})</td></tr>
        <tr><th>Matched TF-IDF Features</th><td>{res['matched_features']}</td></tr>
        <tr><th>Processed Tokens</th><td>{res['token_count']}</td></tr>
        <tr><th>SVM Decision Margin</th><td>{raw_display}</td></tr>
    </tbody>
</table>

<div style="font-family:'IBM Plex Mono', monospace; font-size:11px; text-transform:uppercase; letter-spacing:0.06em; color:#57534e;">
    Cross-Check Trusted Sources
</div>
<div class="sources-row">{sources_html}</div>

<div class="disclaimer">
    Disclaimer: Predictions are generated by an automated NLP/ML model (TF-IDF + Linear SVM)
    and are NOT verified fact-checks. Always cross-check important claims with trusted sources.
</div>
""",
            unsafe_allow_html=True,
        )
