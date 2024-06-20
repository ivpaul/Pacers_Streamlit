import logging
import streamlit as st
import datetime


# Initialize logging
def init_logging():
    logger = logging.getLogger("streamlit_app")
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler("log.txt")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def log_access():
    logger = logging.getLogger("streamlit_app")
    logger.info("App accessed")


# Main function to run the Streamlit app
def main():
    logger = logging.getLogger("streamlit_app")
    logger.info("Inside main")

    st.title("Main Page")

    text = st.sidebar.text_input("Text:")
    logger.info(f"User entered text: {text}")

    # URL to be displayed
    url = "https://www.runpacers.com"

    # Display the URL as a clickable link
    st.markdown(f"### [Visit Pacers Website]({url})")


if __name__ == "__main__":
    init_logging()
    log_access()
    main()
