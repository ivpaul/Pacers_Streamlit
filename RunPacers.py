import streamlit as st
import datetime

# Set the page configuration at the very top
st.set_page_config(
    page_title="Main",
    page_icon=":smile:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom logging function
def custom_log(message):
    log_file_path = 'access_log.txt'
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"{message} at: {current_time}\n"
    try:
        with open(log_file_path, "a") as f:
            f.write(log_message)
    except Exception as e:
        st.error(f"Failed to log access: {e}")

# Log access every time the app is accessed or used
custom_log("App accessed")

# Streamlit title
st.title("Main Page")

# URL to be displayed
url = "https://www.runpacers.com"

# Display the URL as a clickable link
st.markdown(f"### [Visit Pacers Website]({url})")
