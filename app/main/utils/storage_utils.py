from google.cloud import storage
from .logger import write_cloud_logger
import os

# Configure this environment variable via app.yaml
CLOUD_STORAGE_BUCKET = os.environ['CLOUD_STORAGE_BUCKET']
#CLOUD_STORAGE_BUCKET = 'art-retrieval-api-234614.appspot.com'


def get_file_from_cloud_storage(filename):
    
    # Create a Cloud Storage client.
    gcs = storage.Client()

    # https://console.cloud.google.com/storage/browser/[bucket-id]/
    bucket = gcs.get_bucket(CLOUD_STORAGE_BUCKET)
    # # get bucket data as blob
    blob = bucket.get_blob(os.path.basename(filename))
    blob.download_to_filename(filename)
    write_cloud_logger("File downloaded:" + filename)



def upload_blob(filename, filestr, content_type):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(CLOUD_STORAGE_BUCKET)
    blob = bucket.blob(filename)
    
    blob.upload_from_string(
            filestr,
            content_type= content_type
            )

    write_cloud_logger('File url:' + str(blob.public_url) )
    
    return blob.path