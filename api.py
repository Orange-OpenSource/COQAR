import os
import sys
import argparse
import importlib
import logging
import logzero
from logzero import logger
from server import create_app, Config

def run_server(app,host,port,verbose):
    app.run(host=host,port=port,debug=verbose>=3,threaded=False)

def main():
    parser = argparse.ArgumentParser(description='Question Rewriting API')
    parser.add_argument('--host',default='0.0.0.0',
                        help='Interface to listen to')
    parser.add_argument('--port',default='8080',
                        help='Port number to listen to')
    parser.add_argument('--static-folder',default=None,
                        help='Folder to serve as static content')
    parser.add_argument('--model',default=None,
                        help='URI or path for model to use')
    parser.add_argument('--debug','-d',default=False,action='store_true',
                        help='Enable flask debugging')
    args = parser.parse_args()
    if args.model is not None:
        os.environ['MODEL_URL'] = args.model
    app = create_app({
        'STATIC_FOLDER': args.static_folder
    })
    app.run(host=args.host,port=args.port,debug=args.debug,threaded=False)

if __name__=='__main__':
    main()
