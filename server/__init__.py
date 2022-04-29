import os

from flask import Flask, redirect
from flask_cors import CORS

import logzero
from logzero import logger

from .config import Config


from pkg_resources import get_distribution, DistributionNotFound

__title__ = "qrew"
__version__ = "1.0.0"

def create_app(cfg={}):
    from . import routes
    from . import swagger
    from . import schemas
    app = Flask(__name__,static_url_path='')
    app.config.from_object(Config)
    app.config.update(cfg)
    CORS(app)
    schemas.init_app(app)
    routes.init_app(app)

    logger.info("Configuration: %s"%(", ".join(["%s: %s"%(k,v) for k,v in app.config.items()])))

    @app.route('/',methods=['GET'])
    def entry():
        return app.send_static_file('index.html')

    @app.after_request
    def set_response_headers(response):  # pylint: disable=W0612
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response

    app.register_blueprint(routes.api, url_prefix='/api')
    swagger.init_app(app)
    return app
