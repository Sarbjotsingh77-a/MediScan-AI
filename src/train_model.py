# 1. Import Libraries ( Required for the Project)
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Import feature engineering function
from src.feature_engineering import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer

# 2. Loading of the dataset (samples dataset)
df = pd.read_csv("data/mtsamples.csv", encoding='latin1')

# 3. Basic Cleaning of the Dataset 
df = df.drop(columns=['Unnamed: 0'], errors='ignore')
df = df[['transcription', 'medical_specialty']]
df.dropna(inplace=True)

# 4. Doing the Preprocess of the text 
df['clean_text'] = df['transcription'].apply(clean_text)

# 5. Tf-Idf Feature Extraction for further process
tfidf = TfidfVectorizer(
    max_features=3000,   # reduced for better performance
    ngram_range=(1,2),
    stop_words='english'
)

# Reduce number of classes (New improvement done)
top_classes = df['medical_specialty'].value_counts().nlargest(10).index
df = df[df['medical_specialty'].isin(top_classes)]

X = tfidf.fit_transform(df['clean_text'])
y = df['medical_specialty']

# 6. Training and Testing Spliting part of the Model ( following 80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Training of the Model 

# Logistic Regression
lr_model = LogisticRegression(max_iter=200, class_weight = 'balanced')
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# 8. Comparing the model i.e. LR and RF 
print("Logistic Regression Accuracy:", lr_acc)

# Classification Report (VERY IMP)
print("\nClassification Report:\n")
print(classification_report(y_test, lr_pred))

# 9. Selecting the best model
best_model = lr_model
print("Selected Model: Logistic Regression")

# 10. Save model & Vectorizer 

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("Model and Vectorizer saved successfully!")
