from flask import Response
from werkzeug.exceptions import HTTPException
from marshmallow import ValidationError
from logzero import logger
import json

def auto_abort(f):
    def _wrapper(*args,**kwargs):
        try:
            return f(*args,**kwargs)
        except HTTPException as err:
            logger.exception(err)
            raise
        except ValidationError as err:
            logger.exception(err)
            abort(400, error="Validation failed: %s" % err.messages)
        except Exception as err:  # pylint: disable=W0703
            logger.exception(err)
            abort(500, error="%s" % err)
    _wrapper.__name__ = f.__name__
    _wrapper.__doc__ = f.__doc__
    if _wrapper.__doc__ is not None:
        _wrapper.__doc__ += """
       400:
         description: Validation Error
         schema:
          $ref: '#/definitions/Error'
       500:
         description: Internal Server Error
         schema:
          $ref: '#/definitions/Error'
"""
    return _wrapper

def abort(code, **kwargs):
    description = json.dumps(kwargs)
    response = Response(status=code, mimetype='application/json',
                        response=description)
    raise HTTPException(description=description, response=response)
