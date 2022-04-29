#from flasgger import Swagger, APISpec

from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from apispec_webframeworks.flask import FlaskPlugin
from flask import jsonify
from flask_swagger_ui import get_swaggerui_blueprint

from . import schemas
from . import routes
from . import logger
from . import __title__, __version__

spec = APISpec(
    title=__title__,
    version=__version__,
    openapi_version='2.0.0',
    plugins=[
        FlaskPlugin(),
        MarshmallowPlugin(),
    ],
)

# Schemas definitions
spec.components.schema('Error',schema=schemas.ErrorSchema)
spec.components.schema('Status',schema=schemas.StatusSchema)
spec.components.schema('Input',schema=schemas.InputSchema)
spec.components.schema('Output',schema=schemas.OutputSchema)
spec.components.schema('BatchInput',schema=schemas.BatchInputSchema)
spec.components.schema('BatchOutput',schema=schemas.BatchOutputSchema)

def init_app(app):
    swagger_ui_url = app.config.get('SWAGGER_UI_URL','/apidocs')
    swagger_spec_url = app.config.get('SWAGGER_SPEC_URL','/apidocs/swagger.json')
    with app.test_request_context():
        # register all swagger documented functions here
        for fn_name in app.view_functions:
            if fn_name == 'static':
                continue
            logger.debug(f"Loading swagger docs for function: {fn_name}")
            view_fn = app.view_functions[fn_name]
            spec.path(view=view_fn)
    @app.route(swagger_spec_url)
    def create_swagger_spec():
        return jsonify(spec.to_dict())
    # Call factory function to create our blueprint
    swaggerui_blueprint = get_swaggerui_blueprint(
        swagger_ui_url,
        swagger_spec_url,
        config={
            'app_name': __title__
        }
    )
    app.register_blueprint(swaggerui_blueprint)
