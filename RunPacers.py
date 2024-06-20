import streamlit as st
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# Function to log access
def log_access():
    logging.info('App accessed')

# Call log_access when the app is loaded
log_access()

# Your Streamlit app code
st.title('My Streamlit App')
st.write('This is a sample app with logging.')

# Add more logging if necessary
logging.info('Main page displayed')
