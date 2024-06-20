import streamlit as st
from utils.log import get_logger

logger = get_logger()

logger.info("App accessed")

text = st.sidebar.text_input("Text:")
logger.info(f"User entered text: {text}")

# URL to be displayed
url = "https://www.runpacers.com"

# Display the URL as a clickable link
st.markdown(f"### [Visit Pacers Website]({url})")

