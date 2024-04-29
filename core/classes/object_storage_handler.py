#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import torch
import os
import datetime
from enum import Enum
from pathlib import Path
from ...core.helper_tools import vis_dict, formatException
from ...core.file_system_tools import scan_directory

from typing import Tuple

class ObjectNotLoadedError(Exception): ...
class PathDoesNotExistError(Exception): ...
class UnknownSaverError(Exception): ...

class Saver(Enum):
    NUMPY           = 0
    NUMPY_COMPRESS  = 1
    TORCH           = 2
    TORCH_COMPRESS  = 3

class Object_Storage_Handler():
    def __init__(self, storage_path: Path, build_dir: bool = False, build_dir_parents: bool = False,
                 prefix: str = "obj", saver: Saver = Saver.NUMPY, verbose: bool = False):
        '''
        Initialisation

        Inputs:
        - storage_path:         pathlib.Path type; path to directory to store objects
        - build_dir:            bool type (default: False); whether to build storage directory if it doesn't exist
        - build_dir_parents:    bool type (default: False); whether to build storage directory parent path if it doesn't exist
        - compress:             bool type (default: True); whether to compress stored objects
        - prefix:               str type (default: "obj"); prefix of stored object file names 
        - use_torch:            bool type (default: False); Whether object contains tensors; swaps to torch.load and torch.save 
        Returns:
            Object_Storage_Handler type; self
        '''
        self.verbose = verbose
        resolved_storage_path = storage_path.resolve()

        if build_dir: resolved_storage_path.mkdir(parents=build_dir_parents, exist_ok=True)

        assert os.path.exists(resolved_storage_path), "Directory specified by storage_path does not exist (%s)." % resolved_storage_path
        assert os.path.isdir(resolved_storage_path), "Directory specified by storage_path is not a valid directory (%s)." % resolved_storage_path

        self.storage_path: Path         = resolved_storage_path
        self.param_path: Path           = self.storage_path / "params"

        self.param_path.mkdir(parents=False, exist_ok=True)

        self.set_loader_saver(saver=saver)

        self.prefix: str                = prefix
        self.loaded: bool               = False
        self.saved: bool                = False
        self.stored_object: dict        = {'object': None, 'params': {}}

    def is_loaded(self) -> bool:
        '''
        Check if an object is loaded

        Inputs:
        - None
        Returns:
            bool type; True if loaded
        '''
        return self.loaded

    def get(self, check: bool = False) -> dict:
        '''
        Get loaded stored object container dictionary

        Inputs:
        - check:    bool type (default: False); check if object is loaded and, if it is not loaded, raise ObjectNotLoadedError
        Returns:
            dict type; stored object container dictionary
        '''
        if check and (not self.loaded): raise ObjectNotLoadedError()
        return self.stored_object

    def get_object(self, check: bool = False) -> object:
        '''
        Get loaded stored object

        Inputs:
        - check:    bool type (default: False); check if object is loaded and, if it is not loaded, raise ObjectNotLoadedError
        Returns:
            dict type; stored object
        '''
        if check and (not self.loaded): raise ObjectNotLoadedError()
        return self.stored_object['object']
    
    def set_object(self, object_to_store: object, object_params: dict, saved: bool = False) -> Object_Storage_Handler:
        '''
        Set the stored object and provide describing parameters

        Inputs:
        - object_to_store:  object type; what object to store
        - object_params:    dict type; parameters to describe the object (used on retrieval)
        - saved:            bool type (default: False); whether this object already exists in the stored library
        Returns:
            Object_Storage_Handler type; self
        '''
        self.stored_object['object'] = object_to_store
        self.stored_object['params'] = object_params
        self.loaded = True
        self.saved = saved
        return self

    def get_params(self, check: bool = False) -> dict:
        '''
        Get loaded stored object parameters dictionary

        Inputs:
        - check:    bool type (default: False); check if object is loaded and, if it is not loaded, raise ObjectNotLoadedError
        Returns:
            dict type; stored object parameters dictionary
        '''
        if check and (not self.loaded): raise ObjectNotLoadedError()
        return self.stored_object['params']
    
    def show(self, printer = print) -> str:
        '''
        Print loaded stored object contents

        Inputs:
        - printer:    method type {default: print}; will display contents through provided print method
        Returns:
            str type; representation of stored object dictionary contents
        '''
        return vis_dict(self.stored_object, printer=printer)

    def unload(self) -> Object_Storage_Handler:
        '''
        Unload any stored object

        Inputs:
        - None
        Returns:
            Object_Storage_Handler type; self
        '''
        self.stored_object.pop("object")
        self.stored_object.pop("params")
        self.stored_object["object"] = None
        self.stored_object["params"] = {}
        self.loaded = False
        self.saved = False
        return self
    
    def set_loader_saver(self, saver: Saver) -> Object_Storage_Handler:
        '''
        Set saver/loader combination

        Inputs:
        - saver:     object_storage_handler.Saver enum type
        Returns:
            Object_Storage_Handler type; self
        '''
        if saver in [Saver.TORCH, Saver.TORCH_COMPRESS]:
            compress = saver == Saver.TORCH_COMPRESS
            self.saver  = lambda file_name: self._torch_save(file_name=file_name, compress=compress)
            self.loader = self._torch_load
            self.suffix = ".pt"
        elif saver in [Saver.NUMPY, Saver.NUMPY_COMPRESS]:
            compress = saver == Saver.NUMPY_COMPRESS
            self.saver  = lambda file_name: self._numpy_save(file_name=file_name, compress=compress)
            self.loader = self._numpy_load
            self.suffix = ".npz"
        else:
            raise UnknownSaverError("Saver unknown (%s)." % str(saver))
        return self

    def save(self, overwrite: bool = False) -> Object_Storage_Handler:
        '''
        Save loaded stored object to file system

        Inputs:
        - None
        Returns:
            Object_Storage_Handler type; self
        '''
        
        if (not self.loaded): raise ObjectNotLoadedError()

        file_name = self.params_exist(params=self.get_params())
        if (file_name) and (not overwrite): # file exists, and we aren't allowed to overwrite it
            self.saved = True
            return self
        else:
            if not file_name: # file doesn't exist:
                file_list, _, _, = scan_directory(path=self.param_path, short_files=True)
                name = datetime.datetime.today().strftime(self.prefix + "_%Y%m%d")

                # Generate unique name:
                file_name = name
                count = 0
                while file_name in file_list:
                    file_name = name + "_%d" % count
                    count += 1
            else:
                self._fix(stored_object_base_name=file_name)
        
        return self.saver(file_name=file_name)

    def load(self, params: dict) -> str:
        '''
        Load a stored object container by searching for a param dictionary match

        Inputs:
        - params:   dict type; dictionary matching a stored object's parameters
        Returns:
            str type; file_name if successful, else ''.
        '''
        stored_objects = self._get_possible_parameters()
        for name in stored_objects:
            if stored_objects[name]['params'] == params:
                try:
                    self.loader(file_name=name)
                    return name
                except FileNotFoundError:
                    if self.verbose: print(formatException())
                    self._fix(stored_object_base_name=name)
        return ''

    def params_exist(self, params: dict) -> str:
        '''
        Helper function to check if params already exist in storage

        Inputs:
        - params: dict type; parameter dictionary to compare against in search
        Returns:
            bool type; True if exists
        '''
        stored_objects = self._get_possible_parameters()
        for name in stored_objects:
            if stored_objects[name]['params'] == params:
                return name
        return ''
    
    def _get_possible_parameters(self) -> dict:
        '''
        [Internal] Helper function to iterate over parameter dictionaries and compile a list

        Inputs:
        - None
        Returns:
            dict type; keys correspond to file names in param folder, values are loaded parameter dictionaries
        '''
        
        return {os.path.splitext(entry.name)[0]: 
                    dict(params=dict(np.load(entry.path, allow_pickle=True))['params'].item()) # extract .item(), repackage
                    for entry in os.scandir(self.param_path)
                    if entry.is_file() and entry.name.startswith(self.prefix)}
    
    def _fix(self, stored_object_base_name: str) -> Tuple[bool, bool]:
        '''
        [Internal] Helper function to remove broken/erroneous files

        Inputs:
        - stored_object_name: str type; name of file(s) to purge
        Returns:
            tuple type; contains two booleans: 1) whether a stored object container was purged, 2) whether a param file was purged 
        '''
        stored_object_name          = stored_object_base_name + (self.suffix if not stored_object_base_name.endswith(self.suffix) else '')
        stored_object_param_name    = stored_object_base_name + ('.npz' if not stored_object_base_name.endswith('.npz') else '')
        purged_object               = False
        purged_params               = False
        try:
            os.remove(self.storage_path / stored_object_name)
            if self.verbose: print("Purged: " + str(self.storage_path / stored_object_name))
            purged_object = True
        except FileNotFoundError:
            pass
        try:
            os.remove(self.param_path / stored_object_param_name)
            if self.verbose: print("Purged: " + str(self.param_path / stored_object_param_name))
            purged_params = True
        except FileNotFoundError:
            pass
        return (purged_object, purged_params)
    
    def _torch_save(self, file_name, compress) -> Object_Storage_Handler:
        '''
        [Internal] Helper function to save a stored object to a file name using torch

        Inputs:
        - file_name: str type; name of file to save to
        Returns:
            Object_Storage_Handler type; self
        '''
        full_file_path  = self.storage_path / (file_name + ('.pt' if not file_name.endswith('.pt') else ''))
        full_param_path = self.param_path / (file_name + ('.npz' if not file_name.endswith('.npz') else ''))
        
        np_saver = (np.savez_compressed if compress else np.savez)
        np_saver(full_param_path, params=self.get_params())
        torch.save(obj=self.get_object(), f=full_file_path)

        self.saved = True

        return self

    def _torch_load(self, file_name) -> Object_Storage_Handler:
        '''
        [Internal] Helper function to load a stored object from a file name using torch

        Inputs:
        - file_name: str type; name of file to load
        Returns:
            Object_Storage_Handler type; self
        '''
        full_file_path  = self.storage_path / (file_name + ('.pt' if not file_name.endswith('.pt') else ''))
        full_param_path = self.param_path / (file_name + ('.npz' if not file_name.endswith('.npz') else ''))

        _params = dict(params=dict(np.load(full_param_path, allow_pickle=True))['params'].item())
        _obj    = torch.load(f=full_file_path)
        self.set_object(object_to_store=_obj, object_params=_params)
        return self
    
    def _numpy_save(self, file_name, compress) -> Object_Storage_Handler:
        '''
        [Internal] Helper function to save a stored object to a file name using numpy

        Inputs:
        - file_name: str type; name of file to save to
        Returns:
            Object_Storage_Handler type; self
        '''
        full_file_path  = self.storage_path / file_name
        full_param_path = self.param_path / file_name
        
        np_saver = (np.savez_compressed if compress else np.savez)

        np_saver(full_file_path, **self.get())
        np_saver(full_param_path, params=self.get_params())
        self.saved = True
        return self

    def _numpy_load(self, file_name) -> Object_Storage_Handler:
        '''
        [Internal] Helper function to load a stored object from a file name using numpy

        Inputs:
        - file_name: str type; name of file to load
        Returns:
            Object_Storage_Handler type; self
        '''
        file_name          += '.npz' if not file_name.endswith('.npz') else ''
        full_file_path      = self.storage_path / file_name
        raw_obj = np.load(full_file_path, allow_pickle=True)
        self.set_object(object_to_store=raw_obj['object'].item(), object_params=raw_obj['params'].item(), saved=True)
        return self
    
    def __del__(self):
        '''
        Class destructor
        '''
        self.unload()