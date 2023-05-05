import argparse as ap
import string
from .enum_tools import enum_value_options

# Collection of functions that check whether an input can be appropriately cast
# Functions are named check_(desired output type)

# https://stackoverflow.com/questions/14117415/in-python-using-argparse-allow-only-positive-integers
def check_positive_int(value):
    error_text = "%s is an invalid positive integer value." % (str(value))
    try:
        ivalue = int(value)
    except:
        raise ap.ArgumentTypeError(error_text)
        
    if ivalue <= 0:
        raise ap.ArgumentTypeError(error_text)
    return ivalue

def check_positive_float(value):
    error_text = "%s is an invalid positive float value." % (str(value))
    try:
        ivalue = float(value)
    except:
        raise ap.ArgumentTypeError(error_text)
        
    if ivalue <= 0:
        raise ap.ArgumentTypeError(error_text)
    return ivalue

def check_bounded_float(value, _min, _max, equality):
    error_text = "%s is either an invalid float value." % (str(value))
    error_text_bounds = "%s is not between the required bounds of {%s}."
    try:
        ivalue = float(value)
        if equality == 'lower':
            if not (ivalue >= _min) and (ivalue < _max):
                bounding_str = "%s <= val < %s" % (str(_min), str(_max))
                raise ap.ArgumentTypeError(error_text_bounds % (str(value), bounding_str))
        elif equality == 'upper':
            if not (ivalue > _min) and (ivalue <= _max):
                bounding_str = "%s < val <= %s" % (str(_min), str(_max))
                raise ap.ArgumentTypeError(error_text_bounds % (str(value), bounding_str))
        elif equality == 'both':
            if not (ivalue >= _min) and (ivalue <= _max):
                bounding_str = "%s <= val <= %s" % (str(_min), str(_max))
                raise ap.ArgumentTypeError(error_text_bounds % (str(value), bounding_str))
        elif equality == 'none':
            if not (ivalue > _min) and (ivalue < _max):
                bounding_str = "%s < val < %s" % (str(_min), str(_max))
                raise ap.ArgumentTypeError(error_text_bounds % (str(value), bounding_str))
        else:
            raise Exception('Invalid equality mode %s. Valid modes: lower, upper, both, none.' % str(equality))
    except:
        raise ap.ArgumentTypeError(error_text)
        
    if ivalue <= 0:
        raise ap.ArgumentTypeError(error_text)
    return ivalue

def check_bool(value):
    error_text = "%s is an invalid boolean." % (str(value))
    if isinstance(value, bool): return value
    if isinstance(value, int): return not value == 0
    if isinstance(value, str): return value.lower() == "true"
    raise Exception(error_text)

def check_positive_two_int_tuple(value):
    error_text = "%s is an invalid positive two-integer tuple." % (str(value))
    str_value = str(value) # force to string
    value_list = str_value.replace(' ', '').replace('(','').replace(')','').replace('[','').replace(']','').split(',')
    if not len(value_list) == 2:
        raise ap.ArgumentTypeError(error_text) 
    if '.' in str(str_value):
        raise ap.ArgumentTypeError(error_text) 
    try:
        ivalue = (int(value_list[0]), int(value_list[1]))
    except:
        raise ap.ArgumentTypeError(error_text)
    if not (ivalue[0] > 0 and ivalue[1] > 0):
        raise ap.ArgumentTypeError(error_text)
    return ivalue
    
def check_valid_ip(value):
    error_text = "%s is an invalid ip address." % (str(value))
    ip_raw = str(value)
    if ip_raw == 'localhost':
        return ip_raw
    ip_slice = ip_raw.split('.')
    if not len(ip_slice) == 4:
        raise ap.ArgumentTypeError(error_text)
    for num in ip_slice:
        try:
            int(num)
        except:
            raise ap.ArgumentTypeError(error_text)
    return ip_raw

def check_string(value):
    str_value = str(value)
    if str_value.lower() == 'none':
        return None
    return str_value

def check_string_list(value):
    error_text = "%s is an invalid string list." % (str(value))
    try:
        if isinstance(value, list):
            return [str(i) for i in value]
        return str(value).replace('(','').replace(')','').replace('[','').replace(']','').translate(str.maketrans('', '', string.whitespace)).split(',')
    except:
        raise ap.ArgumentTypeError(error_text)

def check_enum_list(value, enum, skip=[None]):
    error_text = "%s is an invalid (or not accepted) list of identifiers within the enumeration %s" % (str(value), str(enum))
    if isinstance(value, enum):
        value = [value]
    try:
        if not isinstance(value, list):
            value = str(value).replace('(','').replace(')','').replace('[','').replace(']','').translate(str.maketrans('', '', string.whitespace)).split(',')
        all_good = True
        for i in value:
            if not (isinstance(i, enum) and not (i in skip)):
                all_good = False
                break
        if all_good:
            return value # already enum list, done
        
        # get enum information to compare against:
        enum_ids, enum_names = enum_value_options(enum, skip)
        enum_ids_str         = [str(i) for i in enum_ids]
        enum_names_str       = enum_names.replace('(','').replace(')','').replace('[','').replace(']','').translate(str.maketrans('', '', string.whitespace)).split(',')

        # at this point, we should have a list of strings.
        str_list = []
        for i in value:
            if i in enum_ids_str:
                str_list.append(enum[enum_names_str[enum_ids_str.index(i)]])
            elif i in enum_names_str:
                str_list.append(enum[enum_names_str[enum_names_str.index(i)]])
        return str_list
    except:
        pass
    raise ap.ArgumentTypeError(error_text)
        
def check_enum(value, enum, skip=[None]):
    error_text = "%s is an invalid (or not accepted) identifier within the enumeration %s" % (str(value), str(enum))
    if isinstance(value, enum) and not (value in skip):
        return value
    try:
        str_value = str(value)
        enum_ids, enum_names = enum_value_options(enum, skip)
        enum_ids_str = [str(i) for i in enum_ids]
        enum_names_str = enum_names.replace('(','').replace(')','').replace('[','').replace(']','').translate(str.maketrans('', '', string.whitespace)).split(',')
        if str_value in enum_ids_str:
            index = enum_ids_str.index(str_value)
            #print("yes - id", index)
            return enum[enum_names_str[index]]
        if str_value in enum_names_str:
            index = enum_names_str.index(str_value)
            #print("yes - name", index)
            return enum[enum_names_str[index]]
    except:
        pass
    raise ap.ArgumentTypeError(error_text)