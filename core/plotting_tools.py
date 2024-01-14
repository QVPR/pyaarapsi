from typing import List, Tuple

def color_from_value(_val: float, _min: float = 0.0, _max: float = 1.0, inv=False) -> Tuple[float,float,float]:
    '''
    Create RGB triplet from provided float value 
    *** Currently only supports greyscale ***

    Inputs:
    - _val: float type; value to map between _min and _max in colour space
    - _min: float type; minimum value of _val, for normalization
    - _min: float type; maximum value of _val, for normalization
    - inv: bool type {default: False}; whether or not to invert the colour space
    Returns:
        Tuple[float,float,float] bounded between 0 and 1
    '''
    if _val > _max:
        _val = _max
    elif _val < _min:
        _val = _min
    _val_norm = (_val - _min) / (_max - _min)
    if inv:
        _val_norm = 1 - _val_norm
    return (_val_norm,_val_norm,_val_norm)

def colors_from_value(_val: List[float], _min: float = 0.0, _max: float = 1.0, inv=False) -> List[Tuple[float,float,float]]:
    '''
    Create RGB triplets from provided float values 
    *** Currently only supports greyscale ***
    Wrapper for color_from_value

    Inputs:
    - _val: List[float] type; values to map between _min and _max in colour space
    - _min: float type; minimum value of each entry in _val, for normalization
    - _min: float type; maximum value of each entry in _val, for normalization
    - inv: bool type {default: False}; whether or not to invert the colour space
    Returns:
        List[Tuple[float,float,float]] bounded between 0 and 1
    '''
    return [color_from_value(i,_min,_max,inv) for i in _val]