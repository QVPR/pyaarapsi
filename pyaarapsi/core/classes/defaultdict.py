#!/usr/bin/env python3
'''
Dictionary which uses a default key to avoid exceptions. Useful in lambdas.
'''
class DefaultDict(dict):
    '''
    Exactly same as a normal dict, but there >>has<< to be a key called "default". If a get item
    triggers __missing__, returns the value of "default". 
    '''
    def __init__(self, *args, **kwargs):
        assert "default" in kwargs
        super(DefaultDict, self).__init__(*args, **kwargs)

    def __missing__ (self, _):
        return self["default"]
