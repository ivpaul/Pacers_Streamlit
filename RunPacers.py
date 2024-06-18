import streamlit as st
import datetime
import logging

# Set the page configuration at the very top
st.set_page_config(
    page_title="Main",
    page_icon=":smile:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Set up logging
log_file_path = 'access_log.txt'
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Adding file handler without logging format prefix
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)

# Create a custom formatter that only outputs the message
formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)

# Adding handler to the logger
logger.addHandler(file_handler)

# Function to log each access
def log_access():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"App accessed at: {current_time}"
    logger.info(log_message)

# Log access every time the app is accessed or used
try:
    log_access()
except Exception as e:
    logger.error(f"Failed to log access: {e}")
    st.error(f"Failed to log access: {e}")

# Streamlit title
st.title("Main Page")

# URL to be displayed
url = "https://www.runpacers.com"

# Display the URL as a clickable link
st.markdown(f"### [Visit Pacers Website]({url})")
