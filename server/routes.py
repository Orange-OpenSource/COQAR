from flask import Blueprint, request, current_app

from . import utils,schemas, logger, __title__,__version__
from .server import Server

api = Blueprint('api', __name__)

server = Server()

def init_app(app):
    server.init_app(app)

@api.route('/rewrite', methods=['POST'])
@utils.auto_abort
def rewrite():
    """
    Rewrite the provided question according to context
    ---
    post:
     description: Rewrite a question according to context
     parameters:
       - in: body
         name: body
         schema:
          $ref: '#/definitions/Input'
     responses:
       200:
         description: Rewritten data
         schema:
          $ref: '#/definitions/Output'
    """
    data = schemas.InputSchema().load(request.get_json())
    res = server.rewrite(**data)
    return schemas.OutputSchema().jsonify(res)

@api.route('/batch', methods=['POST'])
@utils.auto_abort
def batch_rewrite():
    """
    Rewrite the provided batch of questions.
    Each batch instance is the same as the single rewrite api
    ---
    post:
     description: Rewrite a question batch
     parameters:
       - in: body
         name: body
         schema:
          $ref: '#/definitions/BatchInput'
     responses:
       200:
         description: Rewritten data
         schema:
          $ref: '#/definitions/BatchOutput'
    """
    data = schemas.BatchInputSchema().load(request.get_json())
    res = server.batch_rewrite(instances=data['instances'])
    return schemas.BatchOutputSchema().jsonify(res)


@api.route('/status', methods=['GET'])
@utils.auto_abort
def status():
    """
    Retrieve server status
    ---
    get:
     description: Retrieve server status
     responses:
       200:
         description: Server status
         schema:
          $ref: '#/definitions/Status'
    """
    return schemas.StatusSchema().jsonify({"version": __version__,
                                           "name": __title__})
