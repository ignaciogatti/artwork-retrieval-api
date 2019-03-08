from flask import Flask
from werkzeug.contrib.fixers import ProxyFix
from .config import config_by_name

def create_app(config_name):
    #setup and configuration
    app = Flask(__name__)
    app.wsgi_app = ProxyFix(app.wsgi_app)
    app.config.from_object(config_by_name[config_name])

    from app.main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app 