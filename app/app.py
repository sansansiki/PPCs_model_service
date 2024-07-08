'''
Author: gsl
Date: 2024-01-04 15:06:16
LastEditTime: 2024-01-11 10:01:39
FilePath: /workspace/model_service/app/app.py
Description: main

Copyright (c) 2024 by gsl, All Rights Reserved. 
'''

from factory import create_app
from flask import Flask
from ppc_predict import ppc_bp  
from flask_cors import CORS

app = create_app()
CORS(ppc_bp, supports_credentials=True, resources=r'/*')

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=app.config["PORT"], debug=True)
    # app.run(port=app.config["PORT"], debug=True)
    