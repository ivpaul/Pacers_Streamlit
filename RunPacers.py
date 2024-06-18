import streamlit as st
import datetime
import os

# Function to log each access
def log_access():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"App accessed at: {current_time}\n"
    log_file_path = "access_log.txt"

    try:
        with open(log_file_path, "a") as f:
            f.write(log_message)
        st.write("Logged access successfully.")
    except Exception as e:
        st.error(f"Failed to log access: {e}")


# Log access every time the app is accessed or used
log_access()

# Set the page configuration
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
