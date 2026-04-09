import streamlit as st
from auth_interface import main_auth_interface

# Page configuration
st.set_page_config(
    page_title="ICD-11 Medical Notes Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Run the authentication interface
if __name__ == "__main__":
    main_auth_interface()
