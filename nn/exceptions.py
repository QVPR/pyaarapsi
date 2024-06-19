#! /usr/bin/env python3
'''
Exceptions for better diagnostics
'''

class BadVPRDescriptor(Exception):
    '''
    For wrong FeatureType
    '''

class BadModelClass(Exception):
    '''
    For wrong ModelClass
    '''

class BadSampleMode(Exception):
    '''
    For wrong SampleMode
    '''

class BadGenMode(Exception):
    '''
    For wrong GenMode
    '''

class BadCombosKey(Exception):
    '''
    For bad combos keys
    '''

class BadApplyModel(Exception):
    '''
    For wrong ApplyModel
    '''

class LoadFailure(Exception):
    '''
    For failure in OSH loading.
    '''

class BadScaler(Exception):
    '''
    If a scaler is expected but missing, or improperly initialized
    '''

class BadAssessor(Exception):
    '''
    For wrong assessor
    '''

class BadPredictor(Exception):
    '''
    For wrong predictor
    '''

class BadAblationVersion(Exception):
    '''
    For wrong AblationVersion
    '''
