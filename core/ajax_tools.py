#!/usr/bin/env python3

import numpy as np
from flask import Flask, jsonify, make_response
from flask import request as flask_request
import json
import logging
from enum import Enum
import requests
import time

from .enum_tools import enum_name, enum_get
from .helper_tools import vis_dict
from typing import Union, Type, Tuple

class GET_Method_Types(Enum):
    GET     = 0
    DEL     = 1
    VIEW    = 2
    WEBVIEW = 3
    CHECK   = 4

class POST_Method_Types(Enum):
    NEW     = 0
    ADD     = 1
    APPEND  = 2
    SET     = 3

class AJAX_Connection:
    '''
    Wrapper to communicate with an AJAX database; primary use for Bokeh interfacing
    '''
    def __init__(self, url='http://localhost:5050/data', name='ros'):
        '''
        Initialisation

        Inputs:
        - url:  str type; url of AJAX database
        - name: str type; name to give this instance for identification in requests
        Returns:
        self
        '''
        self.url    = url
        self.name   = name

    def check_if_ready(self):
        try:
            status, _ = self.get(None, GET_Method_Types.CHECK)
            return status
        except:
            return False

    def post(self, key: str, data: dict, method_type: POST_Method_Types, log: bool=False) -> Union[Tuple[dict, requests.Response], Tuple[None, requests.Response]]:
        '''
        Send a POST method AJAX request to the server

        Inputs:
        - key:          str type; name of key in AJAX database to access
        - data:         dict type; data to be actioned in style of :method_type
        - method_type:  POST_Method_Types type; type of POST (method type to action data)
        - log:          bool type; whether to print response contents on receipt
        Returns:
        requests.Response
        '''
        headers = {
                    'Publisher-Name': self.name, 
                    'Method-Type': enum_name(method_type), 
                    'Data-Key': key,
                   }
        response = requests.post(self.url, json={key: data}, headers=headers)
        if log:
            for i in response.__dict__.keys():
                try:
                    print(i + ': ' + str(response.__getattribute__(i)))
                #except AttributeError:
                #    print(i + ': ' + str(response.__getattr__(i)))
                except:
                    print("Couldn't access %s." % i)

        if response._content is None:
            return None, response
        return json.loads(response._content), response

    def get(self, key: Union[str, None], method_type: GET_Method_Types, log: bool=False) -> Union[Tuple[dict, requests.Response], Tuple[None, requests.Response]]:
        '''
        Send a GET method AJAX request to the server

        Inputs:
        - key:          str type; name of key in AJAX database to access
        - method_type:  GET_Method_Types type; type of GET (method type to action on existing data)
        - log:          bool type; whether to print response contents on receipt
        Returns:
        requests.Response
        '''
        headers = {
                    'Publisher-Name': self.name, 
                    'Method-Type': enum_name(method_type), 
                    'Data-Key': str(key),
                   }
        response = requests.get(self.url, headers=headers)
        if log:
            for i in response.__dict__.keys():
                try:
                    print(i + ': ' + str(response.__getattribute__(i)))
                # except AttributeError:
                #     print(i + ': ' + str(response.__getattr__(i)))
                except:
                    print("Couldn't access %s." % i)

        if response._content is None:
            return None, response
        return json.loads(response._content), response

# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)

class Flask_AJAX_Server:
    def __init__(self, port=5050):
        self.app    = Flask(__name__)
        self.data   = {}
        self.port   = port
        self.main()

    def print(self, text):
        print(text)

    def get(self, req_type, req_datakey):
        # Handle data retrieval:
        if req_type == GET_Method_Types.GET:
            if req_datakey in self.data.keys():
                msg = self.data[req_datakey]
            else:
                msg = None
            
        # Handle data deletion:
        elif req_type == GET_Method_Types.DEL:
            msg = not self.data.pop(req_datakey, None) == None
        
        # Handle string overview of database:
        elif req_type == GET_Method_Types.VIEW:
            msg = vis_dict(self.data, printer=None)
        
        # Handle json overview of database:
        elif req_type == GET_Method_Types.WEBVIEW:
            msg = self.data
        
        # Handle check request
        elif req_type == GET_Method_Types.CHECK:
            msg = True
        
        # Otherwise, throw error.
        else:
            raise NotImplementedError("in Flask_AJAX_Server.get, received unknown req_type %s" % str(req_type))
        return jsonify(msg)

    def post(self, req_type, req_datakey, req_data):
        if req_data is None:
            raise TypeError('Data cannot be none for method "POST"')
        
        # Handle database entry creation:
        if req_type == POST_Method_Types.NEW:
            self.data.pop(req_datakey, None)
            self.data[req_datakey] = type(req_data[req_datakey])()
            if isinstance(req_data, dict):
                for _key in req_data[req_datakey].keys():
                    self.data[req_datakey][_key] = type(req_data[req_datakey][_key])()
            return jsonify(self.data[req_datakey])
        
        # Handle database entry setting (overwrite):
        elif req_type == POST_Method_Types.SET:
            self.data.pop(req_datakey, None)
            self.data[req_datakey] = req_data[req_datakey]
            return jsonify(self.data[req_datakey])
        
        # Handle database entry creation and data addition:
        elif req_type == POST_Method_Types.ADD:
            if not req_datakey in self.data.keys():
                self.post(POST_Method_Types.NEW, req_datakey, req_data)
            return self.post(POST_Method_Types.APPEND, req_datakey, req_data)

        # Handle data addiion:
        elif req_type == POST_Method_Types.APPEND:
            if not req_datakey in self.data:
                    raise KeyError('APPEND action called, but %s does not exist in database.' % str(req_datakey))
            if isinstance(self.data[req_datakey], dict):
                self.data[req_datakey].update({
                    k: self.data[req_datakey][k] + req_data[req_datakey][k]
                    for k in req_data[req_datakey].keys()
                })
            elif isinstance(self.data[req_datakey], list):
                self.data[req_datakey].append(req_data)
            else:
                raise Exception('Bad data type [%s is of type %s]' % (req_datakey, str(type(self.data[req_datakey]))))
            return jsonify(self.data[req_datakey])
            
        # Otherwise, throw error.
        else:
            raise NotImplementedError("in Flask_AJAX_Server.post, received unknown req_type %s" % str(req_type))

    def main(self):
        @self.app.route('/data', methods=['GET', 'OPTIONS', 'POST'])
        def generate_response():
            req_method          = flask_request.method
            if req_method in ['GET', 'POST']:
                if not (flask_request.data is ''.encode()):
                    req_data    = json.loads(flask_request.data)
                else:
                    req_data    = None

                try:
                    req_publisher   = flask_request.headers['Publisher-Name']
                    req_datakey     = flask_request.headers['Data-Key']
                    req_methodtype  = flask_request.headers['Method-Type'].upper()

                except KeyError:
                    json_resp       = self.get(GET_Method_Types.WEBVIEW, None)

                else:
                    if req_method == 'GET':
                        req_type    = enum_get(req_methodtype, GET_Method_Types)
                        json_resp   = self.get(req_type, req_datakey)
                    else:
                        req_type    = enum_get(req_methodtype, POST_Method_Types)
                        json_resp   = self.post(req_type, req_datakey, req_data)
                
                resp                = make_response(json_resp)

                    #print('FROM:', req_publisher, '| METHOD:', req_method, '/', req_type, '| DATA:', req_data, '| KEY:', req_datakey)
                    
            else:
                resp                                = make_response()
            h                                       = resp.headers
            h['Access-Control-Allow-Origin']        = '*'
            h['Access-Control-Allow-Methods']       = "GET, OPTIONS, POST"
            h['Access-Control-Max-Age']             = str(21600)
            requested_headers                       = flask_request.headers.get('Access-Control-Request-Headers')
            if requested_headers:
                h['Access-Control-Allow-Headers']   = requested_headers

            return resp

        # show and run
        self.app.run(port=self.port)
