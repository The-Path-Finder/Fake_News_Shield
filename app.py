
import streamlit as st
import joblib

from preprocessing import preprocess_text


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Fake News Shield",
    page_icon="📰",
    layout="wide"
)

# ==========================================
# CUSTOM UI STYLING
# ==========================================

st.markdown("""
<style>

/* Main application background */
.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #172554,
        #1e3a5f
    );
    color: white;
}

/* Main heading */
h1 {
    color: #38bdf8 !important;
    font-size: 42px !important;
    font-weight: 800 !important;
}

/* Subheadings */
h2, h3 {
    color: #e0f2fe !important;
}

/* Normal text */
p, label {
    color: #e2e8f0 !important;
}

/* Text area */
.stTextArea textarea {
    background-color: #f8fafc !important;
    color: #000 !important;
    border: 2px solid #38bdf8 !important;
    border-radius: 12px !important;
    font-size: 16px !important;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(
        90deg,
        #0284c7,
        #2563eb
    );
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 12px 25px;
    font-size: 18px;
    font-weight: bold;
    transition: 0.3s;
}

/* Button hover */
.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #0ea5e9,
        #3b82f6
    );
    border: 1px solid #7dd3fc;
    transform: scale(1.02);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Success message */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Divider */
hr {
    border-color: #38bdf8;
}

/* Footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)



# ==========================================
# 2. LOAD TRAINED MODEL AND VECTORIZER
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

    st.info(
        "Make sure both .pkl files are present "
        "in your project folder."
    )

    st.stop()


# ==========================================
# 3. APPLICATION HEADER
# ==========================================

st.markdown("""
<div style="
    background: linear-gradient(90deg, #075985, #1d4ed8);
    padding: 28px;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 25px;
">
    <h1 style="color: white !important; margin: 0;">
        📰 Fake News Shield
    </h1>
    <p style="color: #e0f2fe !important; font-size: 17px;">
        Intelligent News Classification Using NLP & Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()


# ==========================================
# 4. SIDEBAR
# ==========================================

with st.sidebar:

    st.title("About Project")

    st.write(
        """
        Fake News Shield is an NLP and Machine Learning
        based application designed to classify news
        articles as Fake or Real.
        """
    )

    st.markdown("### Technologies Used")

    st.write("- Python")
    st.write("- NLTK")
    st.write("- TF-IDF")
    st.write("- Linear SVM")
    st.write("- Streamlit")

    st.divider()

    st.caption("Developed as an academic ML/NLP project.")


# ==========================================
# 5. USER INPUT
# ==========================================

st.header("Analyze News Article")

news_text = st.text_area(
    "Enter or paste your news article below:",
    placeholder="Paste your news article here...",
    height=250
)


# ==========================================
# 6. PREDICTION
# ==========================================

if st.button(
    "🔍 Predict News",
    use_container_width=True
):

    if not news_text.strip():

        st.warning("Please enter a news article first.")

    else:

        with st.spinner("Analyzing news article..."):

            try:

                # Step 1: Preprocessing
                cleaned_text = preprocess_text(news_text)

                if not cleaned_text.strip():

                    st.warning(
                        "No meaningful text found after preprocessing."
                    )

                    st.stop()

                # Step 2: TF-IDF Transformation
                transformed_text = tfidf.transform(
                    [cleaned_text]
                )

                # Step 3: ML Model Prediction
                prediction = model.predict(
                    transformed_text
                )[0]

                # Step 4: Display Result
                st.divider()

                st.subheader("Prediction Result")

                # IMPORTANT:
                # Confirm these labels match your Colab encoding.

                if prediction == 0:

                    st.error(
                        "🚨 FAKE NEWS DETECTED"
                    )

                    st.write(
                        "The model classified this article "
                        "as Fake News."
                    )

                elif prediction == 1:

                    st.success(
                        "✅ REAL NEWS"
                    )

                    st.write(
                        "The model classified this article "
                        "as Real News."
                    )

                else:

                    st.warning(
                        f"Unexpected model label: {prediction}"
                    )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )


# ==========================================
# 7. FOOTER
# ==========================================

st.divider()

st.caption(
    "Fake News Shield | NLP + Machine Learning"
)

st.caption(
    "Disclaimer: This application provides ML-based "
    "classification, not independent verification of factual truth."
)
