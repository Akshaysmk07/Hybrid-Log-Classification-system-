import streamlit as st
import pandas as pd
import requests  # To send requests to FastAPI backend
from io import StringIO

# FastAPI backend URL (replace with your Render URL)
FASTAPI_URL = "https://your-app.onrender.com/classify/"

# Streamlit UI
st.title("Log Classification System")

uploaded_file = st.file_uploader("Upload a CSV Log File", type=["csv"])

if uploaded_file:
    st.write("### Uploaded Data Preview")
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    if st.button("Predict"):
        # Convert CSV file to a format suitable for FastAPI
        file_data = {"file": uploaded_file.getvalue()}
        
        # Send file to FastAPI backend
        response = requests.post(FASTAPI_URL, files={"file": uploaded_file})

        if response.status_code == 200:
            # Read the classified CSV from FastAPI response
            classified_df = pd.read_csv(StringIO(response.text))
            st.write("### Classification Results")
            st.dataframe(classified_df)

            # Download button for results
            csv = classified_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", data=csv, file_name="log_predictions.csv", mime="text/csv")
        else:
            st.error(f"Error: {response.json()['detail']}")
