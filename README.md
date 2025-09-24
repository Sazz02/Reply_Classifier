# SvaraAI — Reply Classification (Gradio demo)

This Hugging Face Space demonstrates a **reply-classification app** (positive / neutral / negative) with a colorful Gradio UI.  

The models (TF-IDF vectorizer + Logistic Regression + optional LightGBM) were **trained in Google Colab**, exported as `.pkl` files, and then used here to power a web application.

---

## Files to include in this repo
- `app.py` — the Gradio web app (UI + inference).
- `requirements.txt` — Python dependencies for Hugging Face Spaces.
- `models/tfidf_vectorizer.pkl` — TF-IDF vectorizer (from Colab).
- `models/logreg_model.pkl` — Logistic Regression classifier (from Colab).
- `models/lgbm_model.pkl` — optional LightGBM classifier (from Colab).

**Note:** Place `.pkl` files inside the `models/` folder. File names must match these exactly or update `app.py`.

---

## Workflow (How I built this)

1. **Training in Google Colab**  
   - Preprocessed text replies (lowercase, remove special characters).  
   - Trained a `TfidfVectorizer` on the dataset.  
   - Built baseline **Logistic Regression** and **LightGBM** models.  
   - Evaluated accuracy & F1-score, then saved models with `joblib.dump(...)`.  
   - Exported `.pkl` files (`tfidf_vectorizer.pkl`, `logreg_model.pkl`, `lgbm_model.pkl`).  

2. **Deployment as a Web App**  
   - Used **Gradio** to build a colorful UI.  
   - App loads the `.pkl` models and exposes a simple textbox + dropdown for predictions.  
   - Shows per-class probabilities with styled bars (green = positive, orange = neutral, red = negative).  
   - Packaged everything with `requirements.txt` and deployed to **Hugging Face Spaces**.

---

## How to run locally
1. Clone this repo or download the files.  
2. Create a virtual environment and install requirements:
   ```bash
   pip install -r requirements.txt
