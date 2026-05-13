# 1. Importing the libraries
import streamlit as st
import pdfplumber
from src.predict import predict_text
import pandas as pd

# 2. Loading the prescription dataset
pres = pd.read_csv("prescription_dataset.csv")

def suggest_treatment(text):
    text = text.lower()

    for _, row in pres.iterrows():
        disease = str(row['Disease']).lower()
        if disease in text:
            return row['Disease'], row['Medication'], row['Dosage']

    return None, None, None

# 3. Configuration of the page
st.set_page_config(page_title="MediScan AI", layout="centered")

# 4. Title & Header of the Webpage
st.markdown("""
<h1 style='text-align: center; color: #2E86C1;'>🩺 MediScan AI</h1>
<h4 style='text-align: center;'>Smart Medical Report Analyzer</h4>
""", unsafe_allow_html=True)

st.write("---")

# 5. Option for the file Upload
uploaded_file = st.file_uploader("📄 Upload Medical Report (PDF)", type=["pdf"])

# 6. Extraction of the text from the pdfplumber Library
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

# 7. File Processing Part
if uploaded_file:

    with st.spinner("🔍 Extracting and analyzing report..."):
        text = extract_text(uploaded_file)

    st.subheader("📄 Extracted Text (Preview)")
    st.write(text[:400])

    # 8. PREDICTION
    result, confidence = predict_text(text)

    st.success(f"🧠 Predicted Specialty: {result}")
    st.info(f"📊 Confidence Score: {confidence:.2f}%")

    # 9. Suggesting some treatment through dataset
    disease, med, dose = suggest_treatment(text)

    if disease:
        st.info(f"🦠 Possible Disease: {disease}")
        st.write(f"💊 Medication: {med}")
        st.write(f"📏 Dosage: {dose}")
    else:
        st.warning("No direct disease match found in dataset.")

# 10. Option for the Manual Input
st.write("---")
st.subheader("🧪 Test with Manual Input")

user_input = st.text_area("Enter symptoms or medical text:")

if st.button("Predict"):

    if user_input.strip():
        result, confidence = predict_text(user_input)

        st.success(f"🧠 Predicted Specialty: {result}")

        disease, med, dose = suggest_treatment(user_input)

        if disease:
            st.info(f"🦠 Possible Disease: {disease}")
            st.write(f"💊 Medication: {med}")
            st.write(f"📏 Dosage: {dose}")
        else:
            st.warning("No disease match found.")
    else:
        st.error("Please enter some text.")
