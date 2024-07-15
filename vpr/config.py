#!/usr/bin/env python3
'''
Generate config json file
'''

from pathlib import Path
import logging
import json
import os
from typing import Optional

import pyaarapsi.vpr as _root

def make_config(data_path: Path, workspace_path_1: Optional[Path] = None,
                workspace_path_2: Optional[Path] = None, workspace_path_3: Optional[Path] = None):
    '''
    data_path: Path obj;    path to directory where data folder structure will be 
                            generated (if it doesn't already exist)
    '''
    assert isinstance(data_path, Path)

    r_data_path = data_path.resolve()
    r_data_path.mkdir(parents=False, exist_ok=True)
    (r_data_path / 'cfg').mkdir(parents=False, exist_ok=True)
    (r_data_path / 'cfg/svm_models').mkdir(parents=False, exist_ok=True)
    (r_data_path / 'cfg/svm_models/params').mkdir(parents=False, exist_ok=True)
    (r_data_path / 'cfg/svm_models/fields').mkdir(parents=False, exist_ok=True)
    (r_data_path / 'data').mkdir(parents=False, exist_ok=True)
    (r_data_path / 'data/unprocessed_sets').mkdir(parents=False, exist_ok=True)
    (r_data_path / 'data/unprocessed_sets/params').mkdir(parents=False, exist_ok=True)
    (r_data_path / 'data/unprocessed_sets/filt').mkdir(parents=False, exist_ok=True)
    (r_data_path / 'data/unprocessed_sets/filt/params').mkdir(parents=False, exist_ok=True)
    (r_data_path / 'data/compressed_sets').mkdir(parents=False, exist_ok=True)
    (r_data_path / 'data/compressed_sets/params').mkdir(parents=False, exist_ok=True)
    (r_data_path / 'data/compressed_sets/filt').mkdir(parents=False, exist_ok=True)
    (r_data_path / 'data/compressed_sets/filt/params').mkdir(parents=False, exist_ok=True)
    (r_data_path / 'data/image_libraries').mkdir(parents=False, exist_ok=True)
    (r_data_path / 'data/maps').mkdir(parents=False, exist_ok=True)
    (r_data_path / 'data/paths').mkdir(parents=False, exist_ok=True)
    (r_data_path / 'data/rosbags').mkdir(parents=False, exist_ok=True)
    (r_data_path / 'data/videos').mkdir(parents=False, exist_ok=True)
    (r_data_path / 'media').mkdir(parents=False, exist_ok=True)

    _config = dict(data_path=str(r_data_path))

    for _wsp_path, _wsp_name in zip([workspace_path_1, workspace_path_2, workspace_path_3],
                                    ['wsp1', 'wsp2', 'wsp3']):
        if _wsp_path is None:
            continue
        r_wsp = _wsp_path.resolve()
        r_wsp.mkdir(parents=False, exist_ok=True)
        _config[_wsp_name] = str(_wsp_path)

    with open(_root.__path__[0] + '/config.json', 'w', encoding="utf-8") as fp:
        json.dump(_config, fp)

def print_config_error_help():
    '''
    Helper method to print instructions
    '''
    print("\tUnable to read config. Please ensure you have generated a config file.")
    print("\tTo generate a config file, execute:")
    print("\t>>> from pyaarapsi.vpr_simple import config")
    print("\t>>> from pathlib import Path")
    print("\t>>> config.make_config(data_path=Path('/path/to/data/directory/'))")

def get_wsp_path(num: int):
    '''
    Get path to a workspace
    '''
    assert num in [1,2,3]
    try:
        _config = json.load(open(_root.__path__[0] + '/config.json', encoding="utf-8"))
        return _config['wsp' + str(num)]
    except (OSError, json.decoder.JSONDecodeError, AttributeError, TypeError):
        print_config_error_help()
    return None

def get_data_path():
    '''
    Get path to data storage
    '''
    try:
        _config = json.load(open(_root.__path__[0] + '/config.json', encoding="utf-8"))
        return _config['data_path']
    except (OSError, json.decoder.JSONDecodeError, AttributeError, TypeError):
        print_config_error_help()
    return None

def prep_rospkg_root():
    """
    Prepare rospkg root; attempt to automatically find via searching for the aarapsi_robot_pack
    sister package.
    """
    rospkg_root = get_data_path()
    if rospkg_root is None:
        logging.warning("pyaarapsi config file missing. Attempting to build using ROS...")
        try:
            import rospkg #pylint: disable=C0415
            _path = rospkg.RosPack().get_path(rospkg.get_package_name(\
                                                    os.path.abspath(__file__))) + '/'
            make_config(data_path=Path(_path))
            rospkg_root = get_data_path()
            if rospkg_root is None:
                raise rospkg.ResourceNotFound("Failed to get data path.")
        except ImportError:
            logging.warning('Could not access rospkg. This is typically due to a missing or '
                            'incorrect ROS installation.')
        except rospkg.ResourceNotFound:
            logging.warning('Could not find a root path automatically. Ensure you specify root '
                            'argument if using VPRDatasetProcessor.\nAlternatively, see pyaarapsi.'
                            'vpr.config to provide a path for future attempts.')
    return rospkg_root
