import streamlit as st
import datetime

# Function to log each access
def log_access():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("access_log.txt", "a") as f:
        f.write(f"App accessed at: {current_time}\n")

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
