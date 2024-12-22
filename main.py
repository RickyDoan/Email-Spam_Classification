import streamlit as st
import pandas as pd
from joblib import load
from db_helper import text_processing

best_model = load("artifact/best_model.joblib")
vectorizer = load("artifact/vectorizer.joblib")
df = load("artifact/dataframe.joblib")

st.title("Email Spam Classifier")
input_data = st.text_area("Enter the Email or Message for classification",placeholder="Write something here...")


if st.button("Classify"):
    # Ensure the user entered some text
    if input_data.strip():
        # Preprocess the input
        call_predict = text_processing(input_data)
        # Vectorize the input
        vectorize = vectorizer.transform([call_predict])
        # Predict the result
        result = best_model.predict(vectorize)[0]

        # Display the result
        if result == 1:
            st.error("⚠️ This email/message may be Spam! Please handle with care.")
        else:
            st.success("✅ This email/message is not Spam.")
    else:
        st.warning("Please enter an email text to classify.")



required_columns = ["classify", "text"]
if all(col in df.columns for col in required_columns):
    # Prepare options for dropdown: combine 'classify' and 'text' into a single string
    options = [f"{row['classify']} - Text: {row['text']}" for _, row in df.iterrows()]

    # Use Streamlit selectbox for dropdown
    selected_option = st.selectbox("Reference some options of spam or no spam below:", options)

    # Find the selected row using the index of the selected_option
    selected_index = options.index(selected_option)
    selected_row = df.iloc[selected_index]

    # Display details of the selected row
    st.write("You selected this row:")
    st.write(selected_row)
else:
    st.error(f"The DataFrame must contain the following columns: {required_columns}")
