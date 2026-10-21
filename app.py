
import time
import joblib
import streamlit as st
from scipy.sparse import hstack

st.set_page_config(
    page_title="SciDetect | Medical AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 75% 0%, #102b4a 0%, transparent 38%),
        linear-gradient(135deg, #060d1b, #09152a 60%, #0b1024);
    color: #edf5ff;
    font-family: 'Inter', sans-serif;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1450px; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #081329, #0b1930);
    border-right: 1px solid #20395b;
}
section[data-testid="stSidebar"] * { color: #d8e7ff; }

.hero, .panel, .result-card, .metric-card {
    border: 1px solid #24456d;
    border-radius: 20px;
    background: linear-gradient(145deg, rgba(14,31,57,.96), rgba(8,19,38,.96));
}
.hero { padding: 30px; margin-bottom: 22px; }
.panel { padding: 24px; margin-bottom: 20px; }
.result-card {
    padding: 26px;
    border: 1px solid #24d7e8;
    box-shadow: 0 0 28px rgba(37, 190, 238, .09);
}
.eyebrow {
    color: #56e5d0;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.gradient-title {
    background: linear-gradient(90deg, #f4f8ff, #9e91ff, #3bded9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: clamp(29px, 3vw, 43px);
    line-height: 1.2;
    font-weight: 800;
}
.subtitle { color: #9fb5d4; line-height: 1.7; }
.metric-card { padding: 16px; min-height: 100px; }
.metric-label { color: #8ea8ca; font-size: 12px; margin-bottom: 8px; }
.metric-value { font-size: 25px; font-weight: 800; color: #56e5d0; }
.result-label {
    color: #61e8d5; font-size: 12px; font-weight: 800;
    letter-spacing: 1.5px; text-transform: uppercase;
}
.result-name { font-size: 34px; font-weight: 800; color: #f2f7ff; }
.small-note { color: #8ea8ca; font-size: 12px; }
div.stButton > button {
    border: 0;
    border-radius: 12px;
    min-height: 48px;
    font-weight: 700;
    color: white;
    background: linear-gradient(95deg, #803cff, #397bff, #24d8c0);
    transition: .2s ease;
}
div.stButton > button:hover {
    box-shadow: 0 0 22px rgba(71, 151, 255, .32);
    border: 1px solid #6be7e0;
    color: white;
}
div[data-testid="stTextArea"] textarea {
    background: #071226;
    color: #edf5ff;
    border: 1px solid #234a78;
    border-radius: 13px;
    line-height: 1.7;
}
hr { border-color: #20395b; }
</style>
""", unsafe_allow_html=True)


# ---------- Load model files ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("scidetect_svm_final.pkl")
    word_vectorizer = joblib.load("scidetect_word_tfidf_final.pkl")
    char_vectorizer = joblib.load("scidetect_char_tfidf_final.pkl")
    classes = joblib.load("scidetect_classes.pkl")
    return model, word_vectorizer, char_vectorizer, classes


try:
    svm_model, word_tfidf, char_tfidf, classes = load_artifacts()
    model_ready = True
except Exception as exc:
    model_ready = False
    st.error(
        "Could not load the model files. Check that all four .pkl files "
        "are in the same folder as app.py."
    )
    with st.expander("Technical details"):
        st.code(str(exc))


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 22px">
      <div style="font-size:30px;font-weight:800;letter-spacing:-1px">
        🩺 <span style="color:#5bdedb">Sci</span><span style="color:#a58bff">Detect</span>
      </div>
      <div style="color:#8ea8ca;font-size:12px;padding-left:44px">
        MEDICAL NLP INTELLIGENCE
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        ["⌂  Home", "◈  About", "▤  Model Info", "✧  Examples"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**CURRENT MODEL**")
    st.markdown("""
    <div style="background:#0b1c34;border:1px solid #24456d;
                padding:15px;border-radius:14px">
      <div style="color:#55e4c2;font-weight:700">● SVM ENGINE</div>
      <div style="font-size:12px;color:#9fb5d4;margin-top:8px">
        Word + Character TF-IDF<br>
        Accuracy: 78.39%<br>
        Macro F1: 61.34%
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("RESEARCH PROTOTYPE · NOT FOR CLINICAL DECISIONS")


# ---------- Pages ----------
if page.endswith("About"):
    st.markdown('<div class="eyebrow">ABOUT THE PROJECT</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-title">Medical text, better organized.</div>', unsafe_allow_html=True)
    st.markdown("""
    SciDetect is an NLP project that predicts the likely medical specialty
    associated with a clinical transcription.

    **Workflow**
    1. Enter a medical transcription.
    2. Convert it into word and character TF-IDF features.
    3. Use a trained Linear SVM to predict the specialty.
    4. Display the predicted class.

    SciDetect is an educational research prototype, not a diagnostic tool.
    """)

elif page.endswith("Model Info"):
    st.markdown('<div class="eyebrow">MODEL DETAILS</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-title">Inside the prediction engine</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    for col, value, label in [
        (a, "78.39%", "Test accuracy"),
        (b, "61.34%", "Test Macro F1"),
        (c, "77.23%", "Test Weighted F1"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    **Feature extraction:** Word TF-IDF (1–2 grams) combined with character
    TF-IDF (3–5 character n-grams).

    **Classifier:** Linear Support Vector Classification (LinearSVC),
    with `C = 0.5` and balanced class weights.

    **Evaluation note:** The test set was inspected during earlier
    experimentation, so these results are not an untouched final benchmark.
    """)

elif page.endswith("Examples"):
    st.markdown('<div class="eyebrow">TRY AN EXAMPLE</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-title">Explore sample transcriptions</div>', unsafe_allow_html=True)
    examples = {
        "Respiratory symptoms": (
            "The patient reports persistent cough, wheezing and shortness "
            "of breath. Pulmonary examination was performed."
        ),
        "Cardiac assessment": (
            "The patient presents with chest discomfort and palpitations. "
            "An electrocardiogram was obtained for further evaluation."
        ),
        "Orthopedic assessment": (
            "The patient reports knee pain following an injury. Physical "
            "examination assessed range of motion and joint stability."
        ),
    }
    for title, sample in examples.items():
        with st.expander(title):
            st.write(sample)
            if st.button("Use this example", key=title):
                st.session_state["transcription"] = sample
                st.session_state["page_home_notice"] = True
                st.rerun()
    st.info("Example text is illustrative. Predictions may be incorrect.")

else:
    # Hero
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">INTELLIGENT MEDICAL TEXT ANALYSIS</div>
      <div class="gradient-title">
        Turn medical transcriptions<br>into meaningful insights.
      </div>
      <p class="subtitle">
        Analyze clinical language and predict the most likely medical specialty
        using SciDetect's trained NLP model.
      </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, icon, value, label in [
        (c1, "◎", "78.39%", "TEST ACCURACY"),
        (c2, "✳", str(len(classes)), "MEDICAL CLASSES"),
        (c3, "⌘", "Linear SVM", "SELECTED MODEL"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card"><div style="font-size:20px;color:#5bdedb">'
                f'{icon}</div><div class="metric-value">{value}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### ▤  Enter Medical Transcription")
    st.markdown(
        '<p class="subtitle">Paste clinical text below. SciDetect will analyze '
        'the text and predict its likely specialty.</p>',
        unsafe_allow_html=True,
    )

    example_text = (
        "The patient presents with chest discomfort and palpitations. "
        "An electrocardiogram was obtained for further evaluation."
    )

    if st.session_state.pop("page_home_notice", False):
        st.toast("Example loaded.")

    text = st.text_area(
        "Medical transcription",
        key="transcription",
        height=190,
        max_chars=5000,
        placeholder=(
            "e.g. The patient presents with shortness of breath, "
            "wheezing and a history of asthma..."
        ),
        label_visibility="collapsed",
    )

    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        analyze = st.button("✦  Analyze Transcription", use_container_width=True)
    with col_b:
        if st.button("↻  Use Example", use_container_width=True):
            st.session_state["transcription"] = example_text
            st.rerun()
    with col_c:
        if st.button("Clear text", use_container_width=True):
            st.session_state["transcription"] = ""
            st.session_state.pop("last_prediction", None)
            st.rerun()

    st.caption(f"{len(text)} / 5000 characters")
    st.markdown("</div>", unsafe_allow_html=True)

    if analyze:
        if not model_ready:
            st.warning("Resolve the missing model files before analyzing text.")
        elif not text.strip():
            st.warning("Please enter a medical transcription first.")
        else:
            with st.spinner("Analyzing clinical text..."):
                word_features = word_tfidf.transform([text.strip()])
                char_features = char_tfidf.transform([text.strip()])
                features = hstack([word_features, char_features]).tocsr()
                prediction = int(svm_model.predict(features)[0])
                specialty = str(classes[prediction])
                st.session_state["last_prediction"] = specialty

    specialty = st.session_state.get("last_prediction")
    if specialty:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-card">
          <div class="result-label">✦ PREDICTION RESULT</div>
          <div class="result-name">🩺 {specialty}</div>
          <p class="subtitle">
            This is the specialty predicted by the trained SciDetect model
            for the submitted transcription.
          </p>
          <hr>
          <div style="color:#dceaff;font-weight:700">◈ How to interpret this</div>
          <p class="subtitle">
            The prediction is based on learned word and character patterns.
            It is not a calibrated confidence score and may be incorrect,
            especially for rare or overlapping specialties.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="border:1px solid #20395b;border-radius:14px;padding:18px;
                color:#8ea8ca;font-size:12px;text-align:center">
      SCIDETECT &nbsp; · &nbsp; MEDICAL NLP RESEARCH PROJECT
      <br><br>
      Educational use only. Predictions are not medical advice or diagnoses.
    </div>
    """, unsafe_allow_html=True)
