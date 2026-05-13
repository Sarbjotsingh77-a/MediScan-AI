# 1. Importing the Librarires
import pickle
import re

# 2. LOAD MODEL AND VECTORIZER
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# 3. TEXT PREPROCESSING
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# 4. PREDICTION FUNCTION
def predict_text(text):
    # Clean input text
    text = clean_text(text)

    # Convert to TF-IDF features
    features = tfidf.transform([text])

    # Predict using trained model
    prediction = model.predict(features)[0]
    # Confidence score
    proba = model.predict_proba(features)
    confidence = max(proba[0]) * 100
    return prediction, confidence

    

# 5. TEST PREDICTION (OPTIONAL)
if __name__ == "__main__":
    sample_text = "Patient has chest pain and difficulty breathing"
    result = predict_text(sample_text)
    
    print("Sample Input:", sample_text)
    print("Predicted Medical Specialty:", result)