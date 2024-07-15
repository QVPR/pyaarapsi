#!/usr/bin/env python3
'''
AJAX and Flask implementations
'''
import json
from enum import Enum, unique
from typing import Union, Tuple

import requests
from flask import Flask, jsonify, make_response
from flask import request as flask_request

from pyaarapsi.core.enum_tools import enum_name, enum_get
from pyaarapsi.core.helper_tools import vis_dict

DEBUG = False

@unique
class GETMethodTypes(Enum):
    '''
    For GET API
    '''
    GET     = 0
    DEL     = 1
    VIEW    = 2
    WEBVIEW = 3
    CHECK   = 4

@unique
class POSTMethodTypes(Enum):
    '''
    For POST API
    '''
    NEW     = 0
    ADD     = 1
    APPEND  = 2
    SET     = 3

class AJAXConnectionError(Exception):
    '''
    Connection error for AJAX
    '''

class FlaskConnectionError(Exception):
    '''
    Connection error for Flask
    '''

class AJAXDataError(Exception):
    '''
    Data error for AJAX
    '''

class FlaskDataError(Exception):
    '''
    Data error for Flask
    '''

class AJAXConnection:
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
        ''''
        TODO
        '''
        try:
            status, _ = self.get(None, GETMethodTypes.CHECK)
            return status
        except AJAXConnectionError:
            return False

    def post(self, key: str, data: dict, method_type: POSTMethodTypes, log: bool=False
             ) -> Union[Tuple[dict, requests.Response], Tuple[None, requests.Response]]:
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
        try:
            headers = {
                        'Publisher-Name': self.name, 
                        'Method-Type': enum_name(method_type), 
                        'Data-Key': key,
                    }
            response = requests.post(self.url, json={key: data}, headers=headers, timeout=0.5)
            if log:
                for i in response.__dict__.keys():
                    try:
                        print(i + ': ' + str(getattr(response, i)))
                    except AttributeError:
                        print(f"Couldn't access {i}.")
            #pylint: disable=W0212
            if response._content is None:
                return None, response
            return json.loads(response._content), response
            #pylint: enable=W0212
        except Exception as e:
            raise AJAXConnectionError() from e
    #
    def get(self, key: Union[str, None], method_type: GETMethodTypes, log: bool=False
            ) -> Union[Tuple[dict, requests.Response], Tuple[None, requests.Response]]:
        '''
        Send a GET method AJAX request to the server

        Inputs:
        - key:          str type; name of key in AJAX database to access
        - method_type:  GET_Method_Types type; type of GET (method type to action on existing data)
        - log:          bool type; whether to print response contents on receipt
        Returns:
        requests.Response
        '''
        try:
            headers = {
                        'Publisher-Name': self.name, 
                        'Method-Type': enum_name(method_type), 
                        'Data-Key': str(key),
                    }
            response = requests.get(self.url, headers=headers, timeout=0.5)
            if log:
                for i in response.__dict__.keys():
                    try:
                        print(i + ': ' + str(getattr(response, i)))
                    except AttributeError:
                        print(f"Couldn't access {i}.")
            #pylint: disable=W0212
            if response._content is None:
                return None, response
            return json.loads(response._content), response
            #pylint: enable=W0212
        except Exception as e:
            raise AJAXConnectionError() from e

class FlaskAJAXServer:
    '''
    Flask wrapper for AJAX
    '''
    def __init__(self, port=5050):
        self.app    = Flask(__name__)
        self.data   = {}
        self.port   = port
        self.main()

    def print(self, text):
        '''
        Printer
        '''
        print(text)

    def get(self, req_type, req_datakey):
        '''
        Get request
        '''
        # Handle data retrieval:
        if req_type == GETMethodTypes.GET:
            if req_datakey in self.data:
                msg = self.data[req_datakey]
            else:
                msg = None
        # Handle data deletion:
        elif req_type == GETMethodTypes.DEL:
            msg = self.data.pop(req_datakey, None) is not None
        # Handle string overview of database:
        elif req_type == GETMethodTypes.VIEW:
            msg = vis_dict(self.data, printer=None)
        # Handle json overview of database:
        elif req_type == GETMethodTypes.WEBVIEW:
            msg = self.data
        # Handle check request
        elif req_type == GETMethodTypes.CHECK:
            msg = True
        # Otherwise, throw error.
        else:
            raise NotImplementedError("in Flask_AJAX_Server.get, received unknown req_type "
                                      f"{str(req_type)}")
        return jsonify(msg)

    def post(self, req_type, req_datakey, req_data):
        '''
        Post request
        '''
        if req_data is None:
            raise TypeError('Data cannot be none for method "POST"')
        # Handle database entry creation:
        if req_type == POSTMethodTypes.NEW:
            self.data.pop(req_datakey, None)
            self.data[req_datakey] = type(req_data[req_datakey])()
            if isinstance(req_data, dict):
                for _key in req_data[req_datakey].keys():
                    self.data[req_datakey][_key] = type(req_data[req_datakey][_key])()
            return jsonify(self.data[req_datakey])
        # Handle database entry setting (overwrite):
        elif req_type == POSTMethodTypes.SET:
            self.data.pop(req_datakey, None)
            self.data[req_datakey] = req_data[req_datakey]
            return jsonify(self.data[req_datakey])
        # Handle database entry creation and data addition:
        elif req_type == POSTMethodTypes.ADD:
            if not req_datakey in self.data:
                self.post(POSTMethodTypes.NEW, req_datakey, req_data)
            return self.post(POSTMethodTypes.APPEND, req_datakey, req_data)
        # Handle data addiion:
        elif req_type == POSTMethodTypes.APPEND:
            if not req_datakey in self.data:
                raise KeyError(f'APPEND action called, but {str(req_datakey)} does not '
                                   'exist in database.')
            if isinstance(self.data[req_datakey], dict):
                self.data[req_datakey].update({
                    k: self.data[req_datakey][k] + req_data[req_datakey][k]
                    for k in req_data[req_datakey].keys()
                })
            elif isinstance(self.data[req_datakey], list):
                self.data[req_datakey].append(req_data)
            else:
                raise FlaskDataError(f'Bad data type [{req_datakey} is of '
                                f'type {str(type(self.data[req_datakey]))}]')
            return jsonify(self.data[req_datakey])
        # Otherwise, throw error.
        else:
            raise NotImplementedError("in Flask_AJAX_Server.post, received unknown req_type "
                                      f"{str(req_type)}")
    #
    def main(self):
        '''
        Main
        '''
        @self.app.route('/data', methods=['GET', 'OPTIONS', 'POST'])
        def generate_response():
            req_method          = flask_request.method
            if req_method in ['GET', 'POST']:
                if flask_request.data is not ''.encode():
                    req_data = json.loads(flask_request.data)
                else:
                    req_data = None
                try:
                    req_publisher = flask_request.headers['Publisher-Name']
                    req_datakey = flask_request.headers['Data-Key']
                    req_methodtype = flask_request.headers['Method-Type'].upper()
                except KeyError:
                    json_resp = self.get(GETMethodTypes.WEBVIEW, None)
                else:
                    if req_method == 'GET':
                        req_type = enum_get(req_methodtype, GETMethodTypes)
                        json_resp = self.get(req_type, req_datakey)
                    else:
                        req_type = enum_get(req_methodtype, POSTMethodTypes)
                        json_resp = self.post(req_type, req_datakey, req_data)
                resp = make_response(json_resp)
                if DEBUG:
                    print('FROM:', req_publisher, '| METHOD:', req_method, '/', req_type, \
                            '| DATA:', req_data, '| KEY:', req_datakey)
            else:
                resp = make_response()
            h = resp.headers
            h['Access-Control-Allow-Origin'] = '*'
            h['Access-Control-Allow-Methods'] = "GET, OPTIONS, POST"
            h['Access-Control-Max-Age'] = str(21600)
            requested_headers = flask_request.headers.get('Access-Control-Request-Headers')
            if requested_headers:
                h['Access-Control-Allow-Headers'] = requested_headers
            return resp
        # show and run
        self.app.run(port=self.port)
