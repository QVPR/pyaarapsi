#!/usr/bin/env python3
'''
This file provides a collections of functions for file system operations, primarly for Linux
'''

import os
from typing import Optional, Tuple, List
import numpy as np

def scan_directory(path: str, short_files: bool = False) -> Tuple[List[str], List[str], List[str]]:
    '''
    Scan a directory to locate files, directories, and unique file extensions
    '''
    dir_scan = os.scandir(path) # https://docs.python.org/3/library/os.html#os.scandir
    fil_names = []
    dir_names = []
    file_exts = []
    for entry in dir_scan:
        if entry.is_file():
            if entry.name.startswith('.'):
                continue
            if short_files:
                fil_names.append(os.path.splitext(entry.name)[0].lower())
            else:
                fil_names.append(entry.name)
            file_exts.append(os.path.splitext(entry)[-1].lower())
        elif entry.is_dir():
            dir_names.append(entry.name)
        else:
            raise OSError("Unknown file type detected.")
    return fil_names, dir_names, list(np.unique(file_exts))

def get_full_file_name(path: str, short_name: str) -> str:
    '''
    Search a path to identify a file, and report the full name.
    '''
    dir_scan = os.scandir(path)
    for entry in dir_scan:
        if entry.is_file():
            if entry.name.startswith('.'):
                continue
            if os.path.splitext(entry.name)[0].lower() == short_name:
                return entry.name
    return ''

def check_dir_type(path: str, filetype: Optional[str] = None, alltype: bool = False) -> bool:
    '''
    Search a directory to check the type of files within
    Returns false if:
    1. The directory has no files but we want a filetype
    2. The directory has files but we want no files
    3. The directory has no files and we want no files (good), but it has no folders either
            (and we want folders)
    4. We wanted a file and it didn't exist, or there were multiple types and we didn't want that
    '''
    fs, ds, exts = scan_directory(path)
    if filetype == '':
        filetype = None
    if len(fs) > 0 and filetype is not None: # contains a file and we want a file
        if filetype in exts: # we want a file and it exists
            if not alltype: # we don't care if there are other file types present:
                return True
            elif len(exts) == 1: # there is only one and it is only what we want
                return True
        return False # result: 4
    if len(ds) > 0 and filetype is None: # contains a folder and we only want folders
        return True
    return False # result: 1,2,3

def check_structure(root, filetype: Optional[str] = None, alltype: bool = False,
                    skip: Optional[list] = None) -> Tuple[bool, List[str]]:
    '''
    Perform deep search of structure using check_dir_type
    '''
    if skip is None:
        skip = []
    if not check_dir_type(root):  # root must be only directories
        return False, []
    dir_paths = []
    for dir_option in os.scandir(root):
        if (dir_option.name in skip) or (dir_option.path in skip):
            continue
        if not check_dir_type(dir_option.path, filetype=filetype, alltype=alltype):
            return False, []
        dir_paths.append(dir_option.path)
    return True, dir_paths

def find_shared_root(dirs: list):
    '''
    Search through directories to find a common root
    '''
    paths       = []
    lengths     = []
    depth       = 0
    built_root  = ""
    # clean paths, find depth of each
    for dir_option in dirs:
        path = os.path.normpath(dir_option).split(os.sep)
        if path[0] == "":
            path.pop(0)
        paths.append(path)
        lengths.append(len(path))
    # find shared root and depth shared:
    for dir_level in range(min(lengths)):
        path_init = paths[0][dir_level]
        add = True
        for path in paths:
            if path[dir_level] != path_init:
                add = False
                break
        if add:
            built_root += ("/" + paths[0][dir_level])
            depth += 1
    dist = max(lengths) - depth
    return depth, dist, built_root
