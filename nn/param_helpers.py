#! /usr/bin/env python3
'''
Parameter generation helpers
'''
from enum import Enum
from typing import Union, List, Any
import numpy as np

from pyaarapsi.vpr.classes.vprdescriptor import VPRDescriptor
from pyaarapsi.core.enum_tools import enum_get
from pyaarapsi.nn.enums import ModelClass, GenMode

#pylint: disable=C0103
class ParamHolder:
    '''
    For type-checking
    '''
    IGNORED_VARIABLES: List[str] = ['IGNORED_VARIABLES']

def isParamHolder(klass: Any) -> bool:
    '''
    Check if klass is a ParamHolder (or at least has attribute IGNORED_VARIABLES)
    '''
    return (isinstance(klass, ParamHolder) or hasattr(klass, 'IGNORED_VARIABLES'))

def get_paramholder_variables(klass: ParamHolder) -> dict:
    '''
    Grab ParamHolder variables.
    '''
    if not isParamHolder(klass):
        return klass
    klass_keys = (klass.__dict__ if isinstance(klass, type) else klass.__class__.__dict__).keys()
    return {k: get_paramholder_variables(getattr(klass,k))
            for k in klass_keys
            if ((not k.startswith('_')) and (not k in klass.IGNORED_VARIABLES))}

def get_svm_features(descriptor_type: Union[VPRDescriptor, str]) -> List[str]:
    '''
    Get SVM features per VPR descriptor.
    '''
    if isinstance(descriptor_type, Enum):
        name = descriptor_type.name
    elif isinstance(descriptor_type, str):
        name = descriptor_type
    else:
        raise VPRDescriptor.Exception(f"Unknown input: {str(descriptor_type)}")
    #
    if name == VPRDescriptor.NETVLAD.name:
        features  = ["area", "mlows"]
    elif name == VPRDescriptor.SAD.name:
        features  = ["grad", "va"]
    elif name == VPRDescriptor.SALAD.name:
        features  = ["senssum_all", "va"]
    elif name == VPRDescriptor.APGEM.name:
        features = ["rIQR", "va"]
    else:
        raise VPRDescriptor.Exception(f"Unknown Descriptor Type: {name}")
    return list(np.sort(features))

def get_model_params(model_class: ModelClass, num_features: int, num_classes: int, layer_size: int,
                     num_layers: int, dropout: float, generate_mode: GenMode, query_length: int):
    '''
    Model parameter generation
    '''
    if isinstance(model_class, Enum):
        name = model_class.name
    elif isinstance(model_class, str):
        name = model_class
    else:
        raise ModelClass.Exception(f"Unknown input: {str(model_class)}")
    #
    if name == ModelClass.BASIC.name:
        return {"input_ftrs": num_features, "n_classes": num_classes, "layer_size": layer_size,
                    "num_layers": num_layers, "dropout": dropout, "add_desc": "svm replacement"}
    elif name == ModelClass.CUSTOM.name:
        first_layer_size = generate_mode.get_structure_output_size(query_length)
        input_struct = generate_mode.get_structure(query_length)
        hidden_struct = ((first_layer_size, 40, 0.0), (40, 40, 0.0), (40, 40, 0.0), (40, 40, 0.0))
        output_struct = (40, 1, 0.0)
        return {"input_structure": input_struct, "hidden_structure": hidden_struct,
                "output_structure": output_struct, "add_sigmoid": True, 
                "add_desc": "svm replacement"}
    else:
        raise ModelClass.Exception(f"Unknown Model Class {name}")

def get_num_features(mode: GenMode, query_length: int = 0):
    '''
    Safe method for GenMode.get_num_features.
    '''
    if isinstance(mode, GenMode):
        mode = mode.name
    elif not isinstance(mode, str):
        raise GenMode.Exception(f"Unknown mode: {str(mode)}")
    #
    gmode = enum_get(mode, GenMode, wrap=False)   # seems pointless, but it is to debug load/save
                                                    # where object versions can differ
    #
    return gmode.get_num_features(query_length=query_length)

def make_storage_safe(object_in):
    '''
    Convert an object into something that behaves during pickling, across compilation.
    Not exhaustive.
    '''
    if isinstance(object_in, (int,float,str,complex,np.number)):
        return object_in
    elif isinstance(object_in, np.ndarray):
        return make_storage_safe(object_in.tolist())
    elif isinstance(object_in, (list, tuple)):
        return [make_storage_safe(i) for i in object_in]
    elif isinstance(object_in, dict):
        return {i: make_storage_safe(object_in[i]) for i in object_in}
    elif isinstance(object_in, Enum):
        return object_in.name
    elif isParamHolder(object_in):
        return make_storage_safe(get_paramholder_variables(object_in))
    elif callable(getattr(object_in, "save_ready", None)):
        return object_in.save_ready()
    else:
        raise TypeError(f"Unknown type: {str(type(object_in))}, {str(object_in)}")
