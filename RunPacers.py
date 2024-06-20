import streamlit as st
import logging
from datetime import datetime
import os

# Set the log file path
log_file_path = os.path.join(os.path.dirname(__file__), 'app.log')

# Configure logging
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# Function to log access
def log_access():
    logging.info('App accessed')

# Call log_access when the app is loaded
log_access()

# Streamlit app configuration
st.set_page_config(
    page_title="Inventory",
    page_icon=":smile:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Log page configuration setup
logging.info('Page configuration set')

# URL to be displayed
url = "https://www.runpacers.com"

# Display the URL as a clickable link
st.markdown(f"### [Visit Pacers Website]({url})")

# Log URL display
logging.info('Pacers website link displayed')
