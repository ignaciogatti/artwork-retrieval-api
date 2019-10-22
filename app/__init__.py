from flask import Flask
from werkzeug.contrib.fixers import ProxyFix
from .config import config_by_name
import os.path
from .main.utils.storage_utils import get_file_from_cloud_storage
from .main.utils.logger import write_cloud_logger




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


    return app 