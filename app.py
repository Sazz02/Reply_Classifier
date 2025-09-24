# app.py
import os
import re
import joblib
import numpy as np
import gradio as gr

# -------------------------
# Helper: safe-loading
# -------------------------
def try_load(path_options):
    for p in path_options:
        if p is None:
            continue
        if os.path.exists(p):
            try:
                model = joblib.load(p)
                print(f"Loaded: {p}")
                return model, p
            except Exception as e:
                print(f"Failed to load {p}: {e}")
    return None, None

ROOT = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
MODEL_DIR = os.path.join(ROOT, "models")

# try multiple plausible names/locations
tfidf_candidates = [
    os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"),
    os.path.join(MODEL_DIR, "tfidf.pkl"),
    os.path.join(ROOT, "tfidf_vectorizer.pkl"),
    os.path.join(ROOT, "tfidf.joblib"),
    os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"),
    os.path.join(MODEL_DIR, "tfidf.joblib"),
]
logreg_candidates = [
    os.path.join(MODEL_DIR, "logreg_model.pkl"),
    os.path.join(MODEL_DIR, "logreg.pkl"),
    os.path.join(ROOT, "logreg_model.pkl"),
    os.path.join(ROOT, "logreg.pkl"),
    os.path.join(MODEL_DIR, "logreg.joblib"),
]
lgbm_candidates = [
    os.path.join(MODEL_DIR, "lgbm_model.pkl"),
    os.path.join(MODEL_DIR, "lgbm.pkl"),
    os.path.join(ROOT, "lgbm_model.pkl"),
    os.path.join(ROOT, "lgbm.pkl"),
]

tfidf, tfidf_path = try_load(tfidf_candidates)
logreg, logreg_path = try_load(logreg_candidates)
lgbm, lgbm_path = try_load(lgbm_candidates)

DEFAULT_LABELS = ['negative', 'neutral', 'positive']

# -------------------------
# Text preprocessing
# -------------------------
def clean_text(t):
    if t is None:
        return ""
    s = str(t).lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    return s.strip()

# -------------------------
# Prediction logic
# -------------------------
import warnings
warnings.filterwarnings("ignore")

def get_model_classes(model):
    if hasattr(model, "classes_"):
        return list(model.classes_)
    if hasattr(model, "classes"):
        return list(model.classes)
    return DEFAULT_LABELS

def predict_one(text, model_choice="Logistic Regression"):
    text_clean = clean_text(text)
    if not text_clean:
        return {
            "label": "neutral",
            "confidence": 0.0,
            "html": "<i>No text provided</i>",
            "error": None
        }

    if tfidf is None:
        return {"label": None, "confidence": 0.0, "html": "", "error": "Vectorizer (tfidf) not found. Upload tfidf_vectorizer.pkl to models/."}

    X = tfidf.transform([text_clean])

    try:
        if model_choice == "Logistic Regression" and logreg is not None:
            probs = logreg.predict_proba(X)[0]
            classes = get_model_classes(logreg)
        elif model_choice == "LightGBM" and lgbm is not None:
            try:
                probs = lgbm.predict_proba(X)[0]
            except Exception:
                probs = lgbm.predict_proba(X.toarray())[0]
            classes = get_model_classes(lgbm)
        else:
            if logreg is not None:
                probs = logreg.predict_proba(X)[0]; classes = get_model_classes(logreg)
            elif lgbm is not None:
                try: probs = lgbm.predict_proba(X)[0]
                except: probs = lgbm.predict_proba(X.toarray())[0]
                classes = get_model_classes(lgbm)
            else:
                return {"label": None, "confidence": 0.0, "html": "", "error": "No model found. Upload logreg_model.pkl or lgbm_model.pkl to models/."}
    except Exception as e:
        return {"label": None, "confidence": 0.0, "html": "", "error": f"Prediction error: {e}"}

    idx = int(np.argmax(probs))
    label = classes[idx]
    confidence = float(probs[idx])

    colors = {
        'positive': '#16a34a',
        'neutral':  '#f59e0b',
        'negative': '#ef4444'
    }
    bars_html = ""
    for c, p in zip(classes, probs):
        col = colors.get(str(c).lower(), "#3b82f6")
        pct = float(p) * 100.0
        bars_html += f"""
        <div style="display:flex;align-items:center;margin-bottom:8px;">
          <div style="width:95px;font-weight:600;color:#e5e7eb;">{c}</div>
          <div style="flex:1;margin-left:10px;background:#1f2937;border-radius:999px;padding:3px;">
            <div style="width:{pct:.2f}%;background:{col};padding:6px 10px;border-radius:999px;color:white;font-weight:700;text-align:right;">
              {pct:.1f}%
            </div>
          </div>
        </div>
        """

    header_html = f"""
    <div style="display:flex;align-items:center;gap:12px;">
      <div style="font-size:16px;font-weight:700;color:#f3f4f6;">Prediction:</div>
      <div style="padding:6px 12px;border-radius:999px;background:{colors.get(str(label).lower(),'#3b82f6')};color:white;font-weight:800;">
        {label.upper()} ({confidence:.2f})
      </div>
    </div>
    <div style="margin-top:12px;">{bars_html}</div>
    """

    return {"label": label, "confidence": float(confidence), "html": header_html, "error": None}

# -------------------------
# Dark theme CSS
# -------------------------
css = """
body { background: linear-gradient(135deg,#0f172a 0%,#1e293b 100%); color:#e5e7eb; }
.app-card { 
  border-radius: 12px; 
  padding: 18px; 
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.6); 
  background: rgba(31,41,55,0.9);
  color: #f9fafb;
}
.title {
  font-weight: 800;
  font-size: 22px;
  margin-bottom: 6px;
  color: #f9fafb;
}
.subtitle {
  color: #9ca3af;
  margin-bottom: 12px;
}
.gr-button {
  border-radius: 10px;
  padding: 10px 16px;
  font-weight:700;
  background: #3b82f6 !important;
  color: white !important;
}
"""

examples = [
    ["Looking forward to our demo next week! Confirm time please.", "Logistic Regression"],
    ["Not interested at this time, thanks.", "LightGBM"],
    ["Can you share pricing and features?", "Logistic Regression"],
]

with gr.Blocks(css=css, theme=gr.themes.Base()) as demo:
    with gr.Row():
        with gr.Column(scale=2):
            gr.HTML("<div class='app-card'><div class='title'>🌙 SvaraAI — Reply Classifier</div>"
                    "<div class='subtitle'>Classify replies as <b style='color:#16a34a'>positive</b> / <b style='color:#f59e0b'>neutral</b> / <b style='color:#ef4444'>negative</b>.</div>"
                    "</div>")
            inp = gr.Textbox(lines=5, placeholder="Type your reply here...", label="Reply text")
            model_choice = gr.Dropdown(choices=["Logistic Regression", "LightGBM"], value="Logistic Regression", label="Model")
            with gr.Row():
                btn = gr.Button("🚀 Classify", variant="primary")
                clear = gr.Button("🧹 Clear")
            output_label = gr.Markdown(value="**Prediction:** _waiting for input_", label="Result")
            output_html = gr.HTML("<i style='color:#9ca3af;'>Probabilities will appear here</i>")
            error_box = gr.Textbox(interactive=False, visible=False)
            gr.Examples(examples=examples, inputs=[inp, model_choice], label="Try these examples")
        with gr.Column(scale=1):
            gr.HTML("<div class='app-card'><div style='font-weight:800;margin-bottom:8px'>ℹ️ About</div>"
                    "<div style='font-size:13px;color:#d1d5db'>This demo uses a TF-IDF vectorizer and a saved classifier (Logistic Regression / LightGBM). "
                    "Upload your saved pickles to <code>models/</code> as described in README.md.</div></div>")
            stats_md = gr.Markdown("**Model files detected:**<br>"
                                   f"- TF-IDF: `{tfidf_path or 'NOT FOUND'}`  \n"
                                   f"- LogReg: `{logreg_path or 'NOT FOUND'}`  \n"
                                   f"- LGBM: `{lgbm_path or 'NOT FOUND'}`  \n")
            download_note = gr.Markdown("<small style='color:#9ca3af;'>If a model is missing upload it to <code>models/</code> or rename files appropriately.</small>")

    def run_and_format(text, model_choice):
        res = predict_one(text, model_choice)
        if res.get("error"):
            return f"**Error:** {res['error']}", "", gr.update(value=f"<div style='color:#ef4444;font-weight:700'>{res['error']}</div>")
        label = res["label"]
        conf = res["confidence"]
        html = res["html"]
        md = f"**Prediction:** **{label.upper()}** — confidence **{conf:.2f}**"
        return md, str(round(conf, 3)), gr.update(value=html)

    btn.click(run_and_format, inputs=[inp, model_choice], outputs=[output_label, error_box, output_html])
    clear.click(lambda: ("**Prediction:** _waiting for input_", "", gr.update(value="<i style='color:#9ca3af;'>Probabilities will appear here</i>")), [], [output_label, error_box, output_html])

    gr.HTML("<div style='margin-top:18px;color:#9ca3af;font-size:13px'>🌌 Built for the SvaraAI assignment • Upload your model pickles into <code>models/</code></div>")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
