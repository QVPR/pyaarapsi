from enum import Enum

def enum_contains(value, enumtype, wrap=False):
# Return true/false if the value exists within enum
    if not isinstance(value, list):
        for i in enumtype:
            if i.value == value:
                if wrap:
                    return [True]
            return True
    else:
        newvalues = [False] * len(value)
        for ind in range(len(value)):
            for i in enumtype:
                if i.value == value[ind]:
                    newvalues[ind] = True
                    break
        return newvalues
    return False

def enum_get(value, enumtype, wrap=False):
# Return enumtype corresponding to value if it exists (or return None)
    if not isinstance(value, list):
        for i in enumtype:
            if i.value == value or i.name == value:
                if wrap:
                    return [i]
                return i
    else:
        for val in value:
            for i in enumtype:
                if i.value == val or i.name == val:
                    val = i
                    break
        return value
    return None


def enum_value_options(enumtype, skip=[None]):
# Return lists of an enumtype's values and a cleaned string variant for printing purposes
    if isinstance(skip, enumtype):
        skip = [skip]
    options = []
    options_text = []
    for i in enumtype:
        if i in skip: 
            continue
        options.append(i.value)
        options_text.append(i.name)
    return options, str(options_text).replace('\'', '')

def enum_value(enum_in, wrap=False):
    if isinstance(enum_in, list):
        return [i.value for i in enum_in]
    if wrap:
        return [enum_in.value]
    return enum_in.value

def enum_name(enum_in, wrap=False):
    if isinstance(enum_in, list):
        return [i.name for i in enum_in]
    if wrap:
        return [str(enum_in.name)]
    return str(enum_in.name)