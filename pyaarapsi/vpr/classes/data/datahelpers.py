#!/usr/bin/env python3
import os
import numpy as np

def reset_params(root_path):
    param_path = root_path + "/params"
    existing_params = [f for f in os.listdir(param_path) if f.endswith(".npz")]
    for i in existing_params:
        os.remove(param_path + "/" + i)
    npz_files = [f for f in os.listdir(root_path) if f.endswith(".npz")]
    for i in npz_files:
        as_dict = dict(np.load(root_path + "/" + i, allow_pickle=True))
        np.savez(root_path + "/params/" + i, params=as_dict["params"].item())

def search_params(param_path, search_pair=None):
    npz_files = [f for f in os.listdir(param_path) if f.endswith(".npz")]
    for i in npz_files:
        as_dict = dict(np.load(param_path + "/" + i, allow_pickle=True))['params'].item()
        if search_pair is None:
            print(as_dict)
        else:
            if as_dict[search_pair[0]] == search_pair[1]:
                print(as_dict)
    
def get_all_unique(param_path, key):
    npz_files = [f for f in os.listdir(param_path) if f.endswith(".npz")]
    values = []
    for i in npz_files:
        as_dict = dict(np.load(param_path + "/" + i, allow_pickle=True))['params'].item()
        values.append(as_dict[key])
    return np.unique(values)
