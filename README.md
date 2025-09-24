# 🌙 SvaraAI - Reply Classifier

A machine learning web application that classifies email replies as **positive**, **negative**, or **neutral** using TF-IDF vectorization and multiple ML models.

## 🚀 Live Demo

**Try the app live:** [https://huggingface.co/spaces/Sazzz02/SWARA_AI](https://huggingface.co/spaces/Sazzz02/SWARA_AI)

## 📋 Project Overview

This project implements a complete ML pipeline for classifying email replies into three categories:
- **Positive**: Interested in meeting/demo
- **Negative**: Not interested/rejection  
- **Neutral**: Non-committal or irrelevant

### 🧩 Part A - ML/NLP Pipeline Implementation

#### Models Trained:
1. **Baseline Model**: Logistic Regression with TF-IDF
2. **Advanced Model**: LightGBM with TF-IDF
3. **Performance**: Both models achieved ~98% accuracy on test data

#### Dataset Processing:
- Loaded and preprocessed email reply dataset
- Text cleaning (lowercase, special character removal)
- TF-IDF vectorization (max 5000 features, English stopwords removed)
- Train/test split with stratification

#### Model Evaluation:
| Model | Accuracy | Macro F1 | Production Recommendation |
|-------|----------|----------|--------------------------|
| Logistic Regression | 98.59% | 98.59% | ✅ **Recommended** |
| LightGBM | 98.12% | 98.12% | Alternative option |

**Production Choice**: Logistic Regression is recommended for production due to:
- Slightly better performance
- Faster inference time
- More interpretable results
- Lower memory footprint
- Better suited for real-time API responses

### 🧩 Part B - Deployment (Web App)

The application is deployed as a Gradio web interface with:
- Real-time text classification
- Model selection (Logistic Regression / LightGBM)
- Confidence scores and probability visualization
- Dark theme UI with interactive examples

#### API Equivalent:
The web app provides functionality similar to a REST API:
- **Input**: Text string via web interface
- **Output**: Classification label, confidence score, and probability distribution
- **Models**: Choice between Logistic Regression and LightGBM

## 🛠️ Setup Instructions

### Method 1: Run on Hugging Face Spaces (Recommended)
1. Visit: [https://huggingface.co/spaces/Sazzz02/SWARA_AI](https://huggingface.co/spaces/Sazzz02/SWARA_AI)
2. Enter your email reply text
3. Select model (Logistic Regression or LightGBM)
4. Click "🚀 Classify" to get predictions

### Method 2: Local Setup

#### Prerequisites
```bash
Python 3.8+
```

#### Installation
1. Clone the repository:
```bash
git clone https://huggingface.co/spaces/Sazzz02/SWARA_AI
cd SWARA_AI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure model files are in the `models/` directory:
   - `tfidf_vectorizer.pkl`
   - `logreg_model.pkl`
   - `lgbm_model.pkl`

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:7860`

## 📁 Project Structure

```
SWARA_AI/
├── app.py                          # Main Gradio application
├── requirements.txt                # Python dependencies
├── SWARA_AI.ipynb                 # Training notebook (Google Colab)
├── models/                        # Trained model files
│   ├── tfidf_vectorizer.pkl       # TF-IDF vectorizer
│   ├── logreg_model.pkl          # Logistic Regression model
│   └── lgbm_model.pkl            # LightGBM model
├── README.md                      # This file
└── answers.md                     # Short answer responses
```

## 🧠 Model Training Process

The models were trained using Google Colab (`SWARA_AI.ipynb`) with the following pipeline:

1. **Data Loading**: Upload CSV dataset with email replies and labels
2. **Preprocessing**: 
   - Text cleaning and normalization
   - Label standardization (lowercase conversion)
   - Removal of rare labels (< 2 instances)
3. **Feature Engineering**: TF-IDF vectorization with 5000 max features
4. **Model Training**: 
   - Logistic Regression (max_iter=1000)
   - LightGBM with default parameters
5. **Evaluation**: Classification report with precision, recall, F1-score
6. **Model Persistence**: Saved using joblib for deployment

## 🎯 Features

- **Real-time Classification**: Instant prediction on text input
- **Multiple Models**: Compare Logistic Regression vs LightGBM
- **Confidence Visualization**: Interactive probability bars
- **Example Inputs**: Pre-loaded examples for testing
- **Dark Theme UI**: Modern, professional interface
- **Responsive Design**: Works on desktop and mobile
- **Error Handling**: Graceful handling of missing models or invalid inputs

## 📊 Model Performance Details

### Logistic Regression Results:
```
              precision    recall  f1-score   support
    negative       0.99      0.98      0.99       142
     neutral       0.99      0.99      0.99       142
    positive       0.97      0.99      0.98       142
    
    accuracy                           0.99       426
   macro avg       0.99      0.99      0.99       426
weighted avg       0.99      0.99      0.99       426
```

### LightGBM Results:
```
              precision    recall  f1-score   support
    negative       0.98      1.00      0.99       142
     neutral       0.98      0.96      0.97       142
    positive       0.99      0.98      0.98       142
    
    accuracy                           0.98       426
   macro avg       0.98      0.98      0.98       426
weighted avg       0.98      0.98      0.98       426
```

## 🔧 Usage Examples

### Web Interface:
1. **Positive Example**: "Looking forward to our demo next week! Confirm time please."
2. **Negative Example**: "Not interested at this time, thanks."
3. **Neutral Example**: "Can you share pricing and features?"

### Expected Outputs:
- **Label**: Classification result (positive/negative/neutral)
- **Confidence**: Probability score (0.0 - 1.0)
- **Visualization**: Probability distribution across all classes

## 🚀 Deployment

The application is deployed on Hugging Face Spaces using:
- **Framework**: Gradio
- **Runtime**: Python 3.8+
- **Dependencies**: Listed in requirements.txt
- **Models**: Pre-trained and serialized using joblib

## 📝 Dependencies

```txt
gradio>=3.34
scikit-learn>=1.0
joblib
lightgbm
pandas
numpy
```

## 🤝 Contributing

To contribute or modify the models:
1. Train new models using the provided Colab notebook
2. Save models using joblib in the `models/` directory
3. Update the model loading logic in `app.py` if needed
4. Test the web interface locally before deployment

## 📄 License

This project is created for the SvaraAI assignment and demonstrates ML model deployment using modern web frameworks.

---

**Built with**: Python, Scikit-learn, LightGBM, Gradio, Hugging Face Spaces
