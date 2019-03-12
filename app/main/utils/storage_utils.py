from google.cloud import storage
import os

# Configure this environment variable via app.yaml
#CLOUD_STORAGE_BUCKET = os.environ['CLOUD_STORAGE_BUCKET']
CLOUD_STORAGE_BUCKET = 'artwork-data'


def get_file_from_cloud_storage(filename):
    
    # Create a Cloud Storage client.
    gcs = storage.Client()

    # https://console.cloud.google.com/storage/browser/[bucket-id]/
    bucket = gcs.get_bucket(CLOUD_STORAGE_BUCKET)
    # # get bucket data as blob
    blob = bucket.get_blob(os.path.basename(filename))
    blob.download_to_filename(filename)