import logging

def init_logging():
    logger = logging.getLogger("streamlit_app")
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler("log.txt")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

def get_logger():
    return logging.getLogger("streamlit_app")

# Initialize logging when the module is imported
init_logging()
