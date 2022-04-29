import os
from distutils.util import strtobool

class Config:
    # Flask configuration
    TESTING = bool(strtobool(os.getenv('TESTING',"False")))
    DEBUG = bool(strtobool(os.getenv('DEBUG',"False")))
    FLASK_ENV = os.getenv('FLASK_ENV','production')
    SECRET_KEY = os.getenv('SECRET_KEY','GDtfDCFYjD')

