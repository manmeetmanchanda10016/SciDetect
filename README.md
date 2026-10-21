# 🩺 SciDetect — Medical Specialty Classification Using NLP

SciDetect is a medical Natural Language Processing (NLP) project that predicts the most likely medical specialty from clinical transcription text using machine learning and deep learning techniques.

The project explores traditional NLP methods and neural language models, with a final TF-IDF + Linear Support Vector Machine (SVM) classifier integrated into an interactive Streamlit application.

## ✨ Features

- Predicts a medical specialty from clinical transcription text.
- Combines word-level and character-level TF-IDF features.
- Uses a Linear SVM classifier for multiclass prediction.
- Provides an interactive Streamlit interface.
- Compares traditional machine learning models with deep learning approaches.

## 🧠 Final Model

**TF-IDF + Linear SVM**

The final classifier combines word-level and character-level text representations to capture medical terminology, phrases, and useful character patterns.

| Metric | Test Score |
|---|---:|
| Accuracy | 78.39% |
| Macro F1-score | 61.34% |
| Weighted F1-score | 77.23% |

*These results are from the project's exploratory evaluation. The test set was examined during earlier model development, so the scores should not be treated as an untouched benchmark.*

## 🛠️ Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF Vectorization
- Linear SVM
- PyTorch and Transformers for deep learning experiments
- Streamlit
- Joblib

## 📂 Project Structure

```text
SciDetect/
├── app.py
├── requirements.txt
├── scidetect_svm_final.pkl
├── scidetect_word_tfidf_final.pkl
├── scidetect_char_tfidf_final.pkl
├── scidetect_classes.pkl
├── .gitignore
└── README.md
```

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/manmeetmanchanda10016/SciDetect.git
cd SciDetect
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Launch the application

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## 📊 Methodology

1. Clean and prepare clinical transcription text.
2. Encode medical specialty labels.
3. Extract word-level TF-IDF features using unigrams and bigrams.
4. Extract character-level TF-IDF features using character n-grams.
5. Combine both feature representations.
6. Train a class-weighted Linear SVM classifier.
7. Evaluate performance using accuracy, macro F1-score, and weighted F1-score.
8. Integrate the trained model into a Streamlit application.

## ⚠️ Disclaimer

SciDetect is an educational NLP and machine learning project. Its predictions are not medical advice and must not be used for clinical diagnosis, treatment decisions, or patient management.

## 👨‍💻 Author

**Manmeet Manchanda**

GitHub: [@manmeetmanchanda10016](https://github.com/manmeetmanchanda10016)

---

*Built to explore clinical text classification, feature engineering, and applied machine learning.*
