from flask import Flask
from werkzeug.contrib.fixers import ProxyFix
from .config import config_by_name
import os.path
from .main.utils.storage_utils import get_file_from_cloud_storage
from .main.utils.logger import write_cloud_logger


MODEL_DIR = os.path.join(os.getcwd(), 'static/model')
MODEL_PATH = os.path.join(MODEL_DIR, 'denoisy_encoder.h5')
METADATA_FILE_NAME = os.path.join( MODEL_DIR, 'train_mayors_style_encoded_with_url.csv' )
MATRIX_FILE_NAME = os.path.join( MODEL_DIR, 'train_mayors_style_encode.npy' )


def create_app(config_name):
    #setup and configuration
    app = Flask(__name__)
    app.wsgi_app = ProxyFix(app.wsgi_app)
    app.config.from_object(config_by_name[config_name])

    from app.main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

    #Set up cloud logger
    write_cloud_logger('Hello world!')

    #get data model from storage
    #Not necessary in flexible enviroments
    '''
    if not( os.path.isfile( METADATA_FILE_NAME ) ):
        get_file_from_cloud_storage( METADATA_FILE_NAME )
    if not( os.path.isfile( MATRIX_FILE_NAME ) ):
        get_file_from_cloud_storage( MATRIX_FILE_NAME )
    if not( os.path.isfile( MODEL_PATH ) ):
        get_file_from_cloud_storage( MODEL_PATH )
    '''

    return app 