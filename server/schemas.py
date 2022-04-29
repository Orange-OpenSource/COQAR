from marshmallow import fields, validate, pre_load
from flask_marshmallow import Marshmallow

ma = Marshmallow()

def init_app(app):
    ma.init_app(app)

class ErrorSchema(ma.Schema):
    code = ma.Integer(required=True)
    message = ma.String(required=True)

class StatusSchema(ma.Schema):
    name = ma.String(required=True)
    version = ma.String(required=True)

class InputSchema(ma.Schema):
    sentence = ma.String(required=True)
    context = ma.List(ma.String,required=True)

class OutputSchema(ma.Schema):
    sentence = ma.String(required=True)

class BatchInputSchema(ma.Schema):
    instances = ma.Nested("InputSchema",required=True,many=True)

class BatchOutputSchema(ma.Schema):
    outputs = ma.List(ma.String,required=True)
