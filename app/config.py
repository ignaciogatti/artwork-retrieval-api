import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    #secret key for session signing, use following to generate new 
    # >>> import os
    # >>> os.urandom(24)
    SECRET_KEY = "\x94'#\xe8\xdc\xd1I\x8f\x0b+\x053\xd2=Ef\xf4d\xbd&\xcb\xf11\xac"
    DEBUG = False

class DevConfig(Config):
    DEBUG = True
    
class TestConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = False #unit testing forms

class ProdConfig(Config):
    DEBUG = False
    

#map keys to config object
config_by_name = dict(
    dev = DevConfig,
    test = TestConfig,
    prod = ProdConfig
)