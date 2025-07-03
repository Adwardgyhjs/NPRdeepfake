from __future__ import absolute_import


from . import networks,options,data,detect_deepfake

from .networks import base_model,resnet,trainer
from .options import base_options,test_options,train_options
from .data import datasets