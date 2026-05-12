# 🩺 MediScan AI – Medical Report Analyzer

MediScan AI is an AI-powered medical report analysis system built using Natural Language Processing (NLP) and Machine Learning. The application analyzes medical text or uploaded PDF reports to predict the relevant medical specialty and provide intelligent insights such as possible diseases, medications, and confidence scores.

The project uses TF-IDF feature extraction and Logistic Regression for medical text classification. It also integrates a prescription dataset to enhance predictions with disease and medicine suggestions. A user-friendly Streamlit interface allows real-time interaction with the model.

---

# 🚀 Features

- 📄 Upload and analyze medical PDF reports
- 🧠 Predict medical specialty using Machine Learning
- 📊 Display confidence score for predictions
- 💊 Suggest possible disease and medication
- 🔍 Extract and analyze medical text
- 🎨 Interactive Streamlit-based UI
- ⚡ Real-time prediction system

---

# 🛠️ Technologies Used

- Python
- Scikit-learn
- NLP (TF-IDF Vectorization)
- Logistic Regression
- Streamlit
- Pandas
- NumPy
- PDFPlumber

---

# 📁 Project Structure

```text
MediScan-AI/
│
├── data/
│   ├── mtsamples.csv
│   └── prescription_dataset.csv
│
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── src/
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
