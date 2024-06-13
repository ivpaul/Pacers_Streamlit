import streamlit as st

# Set the page configuration
st.set_page_config(
    page_title="Main",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# URL to be displayed
url = "https://www.runpacers.com"

# Display the URL as a clickable link
st.markdown(f"### [Visit Pacers Website]({url})")
