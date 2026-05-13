# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# 2. LOAD DATASETS
mts = pd.read_csv("data/mtsamples.csv", encoding='latin1')
pres = pd.read_csv("data/prescription_dataset.csv")

# 3. CLEAN MTSAMPLES DATA
mts = mts.drop(columns=['Unnamed: 0'], errors='ignore')

# Ensure columns exist safely
mts = mts[['transcription', 'medical_specialty', 'keywords']].copy()

mts.dropna(inplace=True)

# 4. TEXT PREPROCESSING
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

mts['clean_text'] = mts['transcription'].apply(clean_text)

# 5. TF-IDF (MAIN FEATURE)
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words='english'
)

X_tfidf = tfidf.fit_transform(mts['clean_text'])

# 6. KEYWORD FEATURES
mts['keywords_clean'] = mts['keywords'].astype(str).apply(lambda x: x.lower())

important_words = ['pain', 'fever', 'infection', 'cancer', 'diabetes']

for word in important_words:
    mts[word] = mts['keywords_clean'].apply(lambda x: 1 if word in x else 0)

# 7. TEXT FEATURES
mts['text_length'] = mts['clean_text'].apply(len)
mts['word_count'] = mts['clean_text'].apply(lambda x: len(x.split()))

# 8. PRESCRIPTION FEATURES

# Convert categorical to numeric safely
pres['Gender'] = pres['Gender'].map({'Male': 0, 'Female': 1})

# Encode disease & medication
from sklearn.preprocessing import LabelEncoder

le_disease = LabelEncoder()
le_med = LabelEncoder()

pres['Disease_encoded'] = le_disease.fit_transform(pres['Disease'].astype(str))
pres['Medication_encoded'] = le_med.fit_transform(pres['Medication'].astype(str))

# Age feature (ensure numeric)
pres['Age'] = pd.to_numeric(pres['Age'], errors='coerce')

# 9. COMBINE FEATURES (MTS)
keyword_features = mts[important_words].values
numeric_features = mts[['text_length', 'word_count']].values

# Combine all extra features
extra_features = np.hstack((numeric_features, keyword_features))

X_final = hstack((X_tfidf, extra_features))

# 10. TARGET VARIABLE
y = mts['medical_specialty']

# 11. FINAL CHECK
print("MTS Feature Shape:", X_final.shape)
print("Target Shape:", y.shape)

print("\nPrescription Dataset Preview:")
print(pres.head())
