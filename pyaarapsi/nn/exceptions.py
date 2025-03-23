#! /usr/bin/env python3
'''
Exceptions for better diagnostics
'''

class BadCombosKey(Exception):
    '''
    For bad combos keys
    '''

class LoadFailure(Exception):
    '''
    For failure in OSH loading.
    '''
