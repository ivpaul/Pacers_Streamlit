import logging
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit import runtime
import datetime


def get_remote_ip() -> str:
    """Get remote ip."""
    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            return "N/A"

        session_info = runtime.get_instance().get_client(ctx.session_id)
        if session_info is None:
            return "N/A"
    except Exception as e:
        return "N/A"

    return session_info.request.remote_ip


class ContextFilter(logging.Filter):
    def filter(self, record):
        record.user_ip = get_remote_ip()
        return super().filter(record)


def init_logging():
    logger = logging.getLogger("foobar")
    if logger.handlers:  # Logger is already set up, don't set up again
        return
    logger.propagate = False
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s [user_ip=%(user_ip)s] - %(message)s")

    # StreamHandler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.addFilter(ContextFilter())
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # FileHandler for logging to a file
    file_handler = logging.FileHandler("access_log.txt")
    file_handler.setLevel(logging.INFO)
    file_handler.addFilter(ContextFilter())
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def main():
    logger = logging.getLogger("foobar")
    logger.info("Inside main")

    # text = st.sidebar.text_input("Text:")
    # logger.info(f"This is the text: {text}")

    # URL to be displayed
    url = "https://www.runpacers.com"

    # Display the URL as a clickable link
    st.markdown(f"### [Visit Pacers Website]({url})")


if __name__ == "__main__":
    init_logging()
    main()
