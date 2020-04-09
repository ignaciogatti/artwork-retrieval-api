import logging
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler


def write_cloud_logger(text):
    # Instantiates a client
    client = google.cloud.logging.Client(project="art-retrieval-api-234614")
    # Connects the logger to the root logging handler;
    client.setup_logging()
    
    logging.warn(text)