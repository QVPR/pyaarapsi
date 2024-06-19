#! /usr/bin/env python3
'''
VPR Helpers
'''
import copy
from enum import Enum
from typing import Union, Tuple, List

import numpy as np
from numpy.typing import NDArray

from pyaarapsi.vpr_simple.vpr_helpers import FeatureType, SVM_Tolerance_Mode
from pyaarapsi.vpr_simple.vpr_dataset_tool import VPRDatasetProcessor
from pyaarapsi.core.helper_tools import m2m_dist


def make_vpr_dataset_params_subset( ft_type: Union[FeatureType, str], npz_dbp: str, bag_dbp: str,
                                    odom_topic: str, img_topics: str,  sample_rate: str,
                                    img_dims: str, filters: Union[str, dict]) -> dict:
    '''
    Format dictionary for VPRDatasetProcessor; subset, as bag_name key is missing.
    '''
    ft_type = ft_type.name if isinstance(ft_type, Enum) else ft_type
    return {'npz_dbp': npz_dbp, 'bag_dbp': bag_dbp,
            'odom_topic': odom_topic, 'img_topics': img_topics, 'sample_rate': sample_rate,
            'ft_types': [ft_type], 'img_dims': img_dims, 'filters': copy.deepcopy(filters)}

def make_svm_dataset_params_subset(tol_mode: Union[SVM_Tolerance_Mode, str], svm_dbp: str, \
                                   ref_subset: dict, qry_subset: dict) -> dict:
    '''
    Format dictionary for SVMModelProcessor; subset, as some keys are missing.
    '''
    tol_mode = tol_mode.name if isinstance(tol_mode, Enum) else tol_mode
    return {'tol_mode': tol_mode, 'svm_dbp': svm_dbp, \
            'ref_subset': copy.deepcopy(ref_subset), 'qry_subset': copy.deepcopy(qry_subset)}

def make_vpr_dataset_params(env: str, cond: str, set_type: str, subset: dict, combos: dict) -> dict:
    '''
    Format complete dictionary for VPRDatasetProcessor.
    '''
    return {'bag_name': combos[env][cond][set_type], **copy.deepcopy(subset)}

def make_svm_dict(env: str, svm_factors: List[str], subset: dict, combos: dict) -> dict:
    '''
    Format complete dictionary for SVMModelProcessor.
    '''
    prot_subset = copy.deepcopy(subset)
    svm_ref_dict = make_vpr_dataset_params(env=env, cond='SVM', set_type='ref',
        subset=prot_subset['ref_subset'], combos=combos)
    svm_qry_dict = make_vpr_dataset_params(env=env, cond='SVM', set_type='qry',
        subset=prot_subset['qry_subset'], combos=combos)
    svm_svm_dict = {'factors': svm_factors, 'tol_thres': combos[env]['tolerance'],
                        'tol_mode': prot_subset['tol_mode']}
    return {'ref': svm_ref_dict, 'qry': svm_qry_dict, 'svm': svm_svm_dict,
            'npz_dbp': svm_ref_dict['npz_dbp'], 'bag_dbp': svm_ref_dict['bag_dbp'],
            'svm_dbp': prot_subset['svm_dbp']}

def make_vpr_dataset(params: dict, vpr_dp: VPRDatasetProcessor, try_gen: bool=True,
                     verbose: bool = False, crop_to_dataset: bool = True) -> dict:
    '''
    Using params (as generated from make_vpr_dataset_params), load and return a dataset from
    VPRDatasetProcessor. Generation can be controlled with try_gen.
    '''
    name = vpr_dp.load_dataset(dataset_params=params, try_gen=try_gen)
    if verbose:
        print(name)
    if crop_to_dataset:
        dataset = copy.deepcopy(vpr_dp.dataset['dataset'])
    else:
        dataset = copy.deepcopy(vpr_dp.dataset)
    vpr_dp.unload()
    return dataset

def make_load_vpr_dataset(env: str, cond: str, set_type: str, subset: dict, combos: dict,
                          vpr_dp: VPRDatasetProcessor, try_gen: bool = True,
                          verbose: bool = False, crop_to_dataset: bool = True) -> dict:
    '''
    Using arguments for make_vpr_dataset_params, load and return a dataset from
    VPRDatasetProcessor. Generation can be controlled with try_gen.
    '''
    params = make_vpr_dataset_params(env=env, cond=cond, set_type=set_type, \
                                     subset=subset, combos=combos)
    return make_vpr_dataset(params=params, try_gen=try_gen, vpr_dp=vpr_dp,
                            verbose=verbose, crop_to_dataset=crop_to_dataset)

def vpr_match_vect_and_match_ind(arr_1: NDArray, arr_2: NDArray) -> Tuple[NDArray, NDArray]:
    '''
    Generate a match distance vector, and find the match index
    '''
    match_vect     = m2m_dist(arr_1=arr_1, arr_2=arr_2)
    match_ind      = np.argmin(match_vect, axis=0)
    return match_vect, match_ind

def vpr_gt_err(ref_xy: NDArray, match_inds: NDArray, true_inds: NDArray, gt_tolerance: float) \
    -> Tuple[NDArray, NDArray]:
    '''
    Calculate ground truth error and classification using supplied tolerance.
    '''
    #
    gt_err          = np.sqrt(  np.square(ref_xy[:,0][true_inds] - ref_xy[:,0][match_inds]) + \
                                np.square(ref_xy[:,1][true_inds] - ref_xy[:,1][match_inds])  )
    gt_class        = gt_err <= gt_tolerance
    return gt_err, gt_class

def perform_vpr(ref_feats: NDArray, qry_feats: NDArray,
                ref_xy: NDArray, qry_xy: NDArray, tolerance: float) \
            -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    '''
    Use image and spatial features to perform and then assess VPR.
    '''
    #
    assert ref_xy.shape[1] == 2, "ref_gts should only have x,y columns!"
    assert qry_xy.shape[1] == 2, "qry_gts should only have x,y columns!"
    # Perform qry-ref matching via euclidean distance to features (match):
    match_vect, match_ind   = vpr_match_vect_and_match_ind(arr_1=ref_feats, arr_2=qry_feats)
    match_dist              = np.min(match_vect, axis=0)
    # Perform qry-ref matching via euclidean distances to ground-truth position vector (true):
    true_vect, true_ind     = vpr_match_vect_and_match_ind(arr_1=ref_xy, arr_2=qry_xy)
    true_dist               = np.min(true_vect, axis=0)
    # Assess accuracy of feature matching:
    gt_err, gt_yn = vpr_gt_err(ref_xy=ref_xy, match_inds=match_ind, true_inds=true_ind, \
                               gt_tolerance=tolerance)
    return match_vect, match_ind, match_dist, true_vect, true_ind, true_dist, gt_err, gt_yn

def find_dataset_uuid_from_params(params: dict, vpr_dp: VPRDatasetProcessor,
                                  try_gen: bool = False) -> str:
    '''
    Get the UUID of a dataset using params.
    '''
    dataset_params = vpr_dp.get_datasets()
    out_params = copy.deepcopy(params)
    out_name = ''
    for name, dataset_param in dataset_params.items():
        if dataset_param['params'] == out_params:
            return name
    if try_gen:
        vpr_dp.load_dataset(out_params, try_gen=True)
        out_name = find_dataset_uuid_from_params(params=out_params, vpr_dp=vpr_dp, try_gen=False)
    return out_name
