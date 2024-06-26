import streamlit as st
import logging
from datetime import datetime
import os

# Streamlit app configuration
st.set_page_config(
    page_title="Inventory",
    page_icon=":smile:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# URL to be displayed
url = "https://www.runpacers.com"

# Display the URL as a clickable link
st.markdown(f"### [Visit Pacers Website]({url})")

# Log URL display
logging.info('Pacers website link displayed')
