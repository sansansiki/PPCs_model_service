'''
Author: gsl
Date: 2024-01-04 15:21:32
LastEditTime: 2024-01-05 10:27:20
FilePath: /workspace/model_service/app/ppc_predict/__init__.py
Description: 

Copyright (c) 2024 by gsl, All Rights Reserved. 
'''

from flask import  Blueprint

# 初始化示例对象 ppcs
ppc_bp = Blueprint("ppcs",__name__,url_prefix='/ppcs')
from . import predict

