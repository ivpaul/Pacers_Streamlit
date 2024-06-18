import streamlit as st
import datetime

# Set the page configuration at the very top
st.set_page_config(
    page_title="Main",
    page_icon=":smile:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# URL to be displayed
url = "https://www.runpacers.com"

# Display the URL as a clickable link
st.markdown(f"### [Visit Pacers Website]({url})")
