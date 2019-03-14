import logging
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler


def write_cloud_logger(text):
    # Instantiates a client
    client = google.cloud.logging.Client()
    # Connects the logger to the root logging handler;
    client.setup_logging()
    
    logging.warn(text)