from flask import Flask

from .config import config_by_name

def create_app(config_name):
    #setup and configuration
    app = Flask(__name__)
    app.config.from_object(config_by_name[config_name])

    from app.main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app 