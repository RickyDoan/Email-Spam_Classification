import streamlit as st
from joblib import load
from db_helper import text_processing

best_model = load("artifact/best_model.joblib")
vectorizer = load("artifact/vectorizer.joblib")
df = load("artifact/dataframe.joblib")

st.title("Email Spam Classifier")


# Initialize session state to store input data
if "input_data" not in st.session_state:
    st.session_state.input_data = ""  # Initialize input_data key


# Function to clear the input data when the "Delete" button is clicked
def clear_input():
    st.session_state.input_data = ""  # Reset session state for input data


# Text input for the email or message (linked to session state)
input_data = st.text_area(
    "Enter the email or message to classify:",
    value=st.session_state.input_data,  # Use session state to persist input
    key="input_data"  # Keep input data synchronized with session state
)

# Create a two-column layout for the 'Classify' and 'Delete' buttons
col1, col2 = st.columns([4, 1])  # Adjust column widths for alignment
with col1:
    # Place the "Classify" button in the first column
    if st.button("Classify"):  # Trigger classification only when button is clicked
        if st.session_state.input_data.strip():  # Validate non-empty input
            # Preprocess the input
            call_predict = text_processing(st.session_state.input_data)

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
            # Show a warning when the input field is empty or whitespace
            st.warning("Please enter an email or message to classify.")

with col2:
    # Place the "Delete" button in the second column
    st.button("Delete Message", on_click=clear_input)  # Clear button functionality




# Check for the required columns in the DataFrame
required_columns = ["classify", "text"]
if all(col in df.columns for col in required_columns):
    # Prepare options for dropdown: combine 'classify' and 'text' into a single string
    options = [f"{row['classify']} - Text: {row['text']}" for _, row in df.iterrows()]

    # Use Streamlit select box for dropdown
    selected_option = st.selectbox("Reference some options of spam or no spam below:", options)

    # Find the selected row using the index of the selected_option
    selected_index = options.index(selected_option)
    selected_row = df.iloc[selected_index]

    # Display details of the selected row
    st.write("You selected this row:")
    st.write(selected_row)
else:
    st.error(f"The DataFrame must contain the following columns: {required_columns}")


