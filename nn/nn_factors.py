#! /usr/bin/env python3
'''
Factors; GenMode tooling
'''
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from pyaarapsi.vpred.vpred_factors import find_va_factor, find_grad_factor, find_adj_grad_factor, \
    find_va_grad_fusion, find_area_factors, find_under_area_factors, find_peak_factors, \
    find_relative_mean_std, find_relative_percentiles, find_iqr_factors, find_match_distance, \
    find_minima_separation, find_sensitivity, find_sum_factor, find_minima_variation, \
    find_removed_factors, find_mean_median_diff, find_adj_minima_separation, find_adj_sensitivity

from pyaarapsi.nn.enums import GenMode
from pyaarapsi.nn.general_helpers import throw
from pyaarapsi.nn.exceptions import BadGenMode

def make_simple_subcomponents(arr):
    '''
    simple subcomponents
    '''
    subcomponents = np.zeros((GenMode.SIMPLE_COMPONENTS.subcomponent_size, arr.shape[1]))
    subcomponents[0, :]        = np.mean(arr, axis=0)
    subcomponents[1, :]        = np.std(arr, axis=0)
    subcomponents[2:23, :]     = np.percentile(arr, np.linspace(0,100,21).astype(int), axis=0)
    subcomponents[23:28, :]    = np.sort(np.partition(arr, 5, axis=0)[0:5,:],axis=0)
    subcomponents[28:33, :]    = -np.sort(np.partition(-arr, 5, axis=0)[0:5,:],axis=0)
    subcomponents[33, :]       = np.sum(arr,axis=0)
    subcomponents[34, :]       = subcomponents[22,:] - subcomponents[2,:]  # Range
    subcomponents[35, :]       = subcomponents[17,:] - subcomponents[7,:]  # IQR
    subcomponents[36, :]       = subcomponents[0, :] / subcomponents[12,:]  # Mean/Median
    subcomponents[37, :]       = (subcomponents[17, :] - subcomponents[12,:]) \
                                / (subcomponents[12, :] - subcomponents[7,:]) # IQR Skew
    subcomponents[38:40, :]    = np.array(find_minima_variation(arr))
    subcomponents[40:42, :]    = np.array(find_removed_factors(arr))
    subcomponents[42:44, :]    = np.array(find_adj_minima_separation(arr))
    subcomponents[44:46, :]    = np.array(find_adj_sensitivity(arr))
    subcomponents[46, :]       = find_grad_factor(arr)
    subcomponents[47, :]       = find_va_factor(arr)
    return subcomponents

def make_norm_simple_subcomponents(arr):
    '''
    normalized, simple subcomponents.
    '''
    percentiles             = np.percentile(arr, np.linspace(0,100,21).astype(int), axis=0)
    var_range               = percentiles[-1] - percentiles[0]
    subcomponents           = np.zeros((GenMode.NORM_SIMPLE_COMPONENTS.subcomponent_size,
                                        arr.shape[1]))
    subcomponents[0, :]     = np.mean(arr, axis=0) / var_range
    subcomponents[1, :]     = np.std(arr, axis=0) / var_range
    subcomponents[2:23, :]  = percentiles / var_range
    subcomponents[23:28, :] = np.sort(np.partition(arr, 5, axis=0)[0:5,:],axis=0) / var_range
    subcomponents[28:33, :] = -np.sort(np.partition(-arr, 5, axis=0)[0:5,:],axis=0) / var_range
    subcomponents[33, :]    = np.sum(arr,axis=0) / (var_range * arr.shape[0])
    subcomponents[34, :]    = var_range / percentiles[-1]
    subcomponents[35, :]    = (subcomponents[17,:] - subcomponents[7,:]) / var_range
    subcomponents[36, :]    = (subcomponents[0, :] / subcomponents[12,:]) / var_range
    subcomponents[37, :]    = (subcomponents[17, :] - subcomponents[12,:]) \
                                / (subcomponents[17, :] - subcomponents[7,:])
    subcomponents[38:40, :] = np.array(find_minima_variation(arr)) # already normalised
    subcomponents[40:42, :] = np.array(find_removed_factors(arr)) # already normalised
    subcomponents[42:44, :] = np.array(find_adj_minima_separation(arr, norm=True))
    subcomponents[44:46, :] = np.array(find_adj_sensitivity(arr)) # already normalised
    subcomponents[46, :]    = find_grad_factor(arr) / var_range
    subcomponents[47, :]    = find_va_factor(arr) # already normalised
    return subcomponents

def make_complex_subcomponents(arr, match_ind = None):
    '''
    complex subcomponents
    '''
    if match_ind is None:
        match_ind = np.argmin(arr, axis=0)
    elif not hasattr(match_ind, '__iter__'):
        match_ind = np.array([match_ind])
    #
    subcomponents = np.zeros((GenMode.COMPLEX_COMPONENTS.subcomponent_size, arr.shape[1]))
    subcomponents[ 0, :]     = np.array(find_va_factor(S=arr))
    subcomponents[ 1, :]     = np.array(find_grad_factor(S=arr))
    subcomponents[ 2, :]     = np.array(find_adj_grad_factor(S=arr))
    subcomponents[ 3:5, :]   = np.array(find_va_grad_fusion(S=arr))
    subcomponents[ 5:9, :]   = np.array(find_area_factors(S=arr, match_ind=match_ind))
    subcomponents[ 9:13, :]  = np.array(find_under_area_factors(S=arr, match_ind=match_ind))
    subcomponents[ 13:15, :] = np.array(find_peak_factors(S=arr))
    subcomponents[ 15:17, :] = np.array(find_relative_mean_std(S=arr))
    subcomponents[ 17:20, :] = np.array(find_relative_percentiles(S=arr))
    subcomponents[ 20:22, :] = np.array(find_iqr_factors(S=arr))
    subcomponents[ 22:26, :] = np.array(find_match_distance(S=arr, match_ind=match_ind))
    subcomponents[ 26:30, :] = np.array(find_minima_separation(S=arr))
    subcomponents[ 30:34, :] = np.array(find_sensitivity(S=arr))
    subcomponents[ 34:37, :] = np.array(find_sum_factor(S=arr))
    subcomponents[ 37:39, :] = np.array(find_minima_variation(S=arr))
    subcomponents[ 39:41, :] = np.array(find_removed_factors(S=arr))
    subcomponents[ 41:43, :] = np.array(find_mean_median_diff(S=arr))
    return subcomponents

def get_hist(arr: NDArray, ind: int, length: int):
    '''
    stack samples with depth
    '''
    delta_left = ind - length + 1
    left = np.max([0, delta_left])
    diff = left - delta_left
    right_arr = arr[left : ind + 1]
    if not diff:
        return right_arr
    left_arr = np.repeat(arr[left:left+1], diff, axis=0)
    return np.concatenate([left_arr, right_arr])

def get_stack(arr: NDArray, length: int):
    '''
    stack samples appropriately
    '''
    return np.stack([get_hist(arr, i, length) for i in range(len(arr))])

def get_bounds_hist(arr: NDArray, length: int, method: Callable):
    '''
    using stacked history, get min and max
    '''
    arr_stacked = np.array([method(get_stack(arr=arr, length=length)[:,i])
                             for i in range(length)])
    arr_min = np.min(arr_stacked, axis=0)
    arr_max = np.max(arr_stacked, axis=0)
    return arr_min, arr_max

def get_rnge_std_hist(arr: NDArray, length: int, method: Callable):
    '''
    using stacked history, get range and std
    '''
    arr_stacked = np.array([method(get_stack(arr=arr, length=length)[:,i])
                             for i in range(length)])
    arr_min    = np.min(arr_stacked, axis=0)
    arr_max    = np.max(arr_stacked, axis=0)
    arr_rnge   = arr_max - arr_min
    arr_std    = np.std(arr_stacked, axis=0)
    return arr_rnge, arr_std

def get_mean_hist(arr: NDArray, length: int, method: Callable):
    '''
    using stacked history, get mean
    '''
    arr_stacked = np.array([method(get_stack(arr=arr, length=length)[:,i]) for i in range(length)])
    arr_mean   = np.mean(arr_stacked, axis=0)
    return arr_mean

def reshape_dataset_components(*args):
    '''
    ensure dimensionality.
    '''
    return (i if i.ndim == 2 else i[np.newaxis,:] if i.ndim == 1 else throw() for i in args)

def ensure_xy(*args):
    '''
    ensure dimensionality.
    '''
    return (i if i.shape[1] == 2 else i[:,0:2] if i.shape[1] == 3 else throw() for i in args)

def make_components(mode: GenMode, vect: NDArray, ref_feats: NDArray, qry_feats: NDArray,
                    inds: NDArray, query_length: int = 1):
    '''
    Handle all component / neural network statistical feature generation.
    '''
    #
    if isinstance(mode, GenMode):
        mode = mode.name
    elif not isinstance(mode, str):
        raise BadGenMode(f"Unknown mode: {str(mode)}")
    #
    # ref_feats: SAD(64x64), 1000 images -> [1000, 4096]
    # qry_feats: ^^        , 3 queries -> [3, 4096]
    # inds:      ^^        , 3 queries -> [3]
    # vect:      ^^        , 1000 ref, 3 query -> [1000, 3]
    #
    dist_mat = vect # Distance Vectors
    ref_feat = np.transpose(ref_feats[inds,:]) # Best matching reference features
    qry_feat = np.transpose(qry_feats) # Query features
    #
    dist_mat = dist_mat[:, np.newaxis] if dist_mat.ndim == 1 else dist_mat
    ref_feat = ref_feat[:, np.newaxis] if ref_feat.ndim == 1 else ref_feat
    qry_feat = qry_feat[:, np.newaxis] if qry_feat.ndim == 1 else qry_feat
    delta_feat = ref_feat - qry_feat
    #
    four_group = [dist_mat, ref_feat, qry_feat, delta_feat]
    #
    # dist_mat -> [1000, 3] # Distance Matrix (Feature Similarity Matrix)
    # ref_feat -> [4096, 3] # Top reference features matched
    # qry_feat -> [4096, 3] # Query features
    # delta_feat -> [4096, 3] # Difference between reference and query features
    #
    # Model -> History of n Query Images
    # repeat for each:
    #   n * (Query features (4096) -> statistics() -> fixed output length (10))
    # [n, 10] -> min() max() along dim of n -> fixed output length (10) * 2 (20 in total)
    # std(), range()
    # [n, 10] -> mean(), nth qry_feat -> fixed output length (10) * 2 (20 in total)
    #
    if mode == GenMode.SIMPLE_COMPONENTS.name:
        comps  = [make_simple_subcomponents(i) for i in four_group]
    elif mode == GenMode.COMPLEX_COMPONENTS.name:
        comps  = [make_complex_subcomponents(i) for i in four_group]
    elif mode == GenMode.NORM_SIMPLE_COMPONENTS.name:
        comps  = [make_norm_simple_subcomponents(i) for i in four_group]
    elif mode == GenMode.TINY_HIST_NORM_SIMPLE_COMPONENTS.name:
        i_0    = np.stack([get_hist(inds, i, query_length) for i in range(len(inds))], axis=1)
        comps  = [make_norm_simple_subcomponents(i) for i in four_group] + [i_0]
    elif mode == GenMode.TINY2_HIST_NORM_SIMPLE_COMPONENTS.name:
        i_0    = np.stack([get_hist(inds, i, query_length) for i in range(len(inds))], axis=1)
        i_1    = np.std(i_0, axis=0)[np.newaxis, :]
        i_2    = np.max(i_0, axis=0)[np.newaxis, :] - np.min(i_0, axis=0)[np.newaxis, :]
        i_3    = np.mean(i_0, axis=0)[np.newaxis, :]
        comps  = [make_norm_simple_subcomponents(i) for i in four_group] + [i_1, i_2, i_3]
    elif mode == GenMode.TINY3_HIST_NORM_SIMPLE_COMPONENTS.name:
        i_0    = np.stack([get_hist(inds, i, query_length) for i in range(len(inds))], axis=1)
        i_1    = np.std(i_0, axis=0)[np.newaxis, :]
        i_2    = np.max(i_0, axis=0)[np.newaxis, :] - np.min(i_0, axis=0)[np.newaxis, :]
        i_3    = np.mean(i_0, axis=0)[np.newaxis, :]
        comps  = [make_norm_simple_subcomponents(i) for i in [dist_mat]] + [i_1, i_2, i_3]
    elif mode == GenMode.HIST_SIMPLE_COMPONENTS.name:
        s_min, s_max = get_bounds_hist(dist_mat, query_length, make_simple_subcomponents)
        d_min, d_max = get_bounds_hist(delta_feat, query_length, make_simple_subcomponents)
        comps  = [s_min, s_max, d_min, d_max]
    elif mode == GenMode.LONG_HIST_SIMPLE_COMPONENTS.name:
        s_min, s_max = get_bounds_hist(dist_mat, query_length, make_simple_subcomponents)
        r_min, r_max = get_bounds_hist(ref_feat, query_length, make_simple_subcomponents)
        q_min, q_max = get_bounds_hist(qry_feat, query_length, make_simple_subcomponents)
        d_min, d_max = get_bounds_hist(delta_feat, query_length, make_simple_subcomponents)
        comps  = [s_min, s_max, r_min, r_max, q_min, q_max, d_min, d_max]
    elif mode == GenMode.HIST_NORM_SIMPLE_COMPONENTS.name:
        s_min, s_max = get_bounds_hist(dist_mat, query_length, make_norm_simple_subcomponents)
        d_min, d_max = get_bounds_hist(delta_feat, query_length, make_norm_simple_subcomponents)
        comps  = [s_min, s_max, d_min, d_max]
    elif mode == GenMode.LONG_HIST_NORM_SIMPLE_COMPONENTS.name:
        s_min, s_max = get_bounds_hist(dist_mat, query_length, make_norm_simple_subcomponents)
        r_min, r_max = get_bounds_hist(ref_feat, query_length, make_norm_simple_subcomponents)
        q_min, q_max = get_bounds_hist(qry_feat, query_length, make_norm_simple_subcomponents)
        d_min, d_max = get_bounds_hist(delta_feat, query_length, make_norm_simple_subcomponents)
        comps  = [s_min, s_max, r_min, r_max, q_min, q_max, d_min, d_max]
    elif mode == GenMode.LONG_HIST2_NORM_SIMPLE_COMPONENTS.name:
        s_1, s_2 = get_rnge_std_hist(dist_mat, query_length, make_norm_simple_subcomponents)
        r_1, r_2 = get_rnge_std_hist(ref_feat, query_length, make_norm_simple_subcomponents)
        q_1, q_2 = get_rnge_std_hist(qry_feat, query_length, make_norm_simple_subcomponents)
        d_1, d_2 = get_rnge_std_hist(delta_feat, query_length, make_norm_simple_subcomponents)
        comps  = [s_1, s_2, r_1, r_2, q_1, q_2, d_1, d_2]
    elif mode == GenMode.LONG_HIST3_NORM_SIMPLE_COMPONENTS.name:
        s_1, s_2 = get_rnge_std_hist(dist_mat, query_length, make_norm_simple_subcomponents)
        r_1, r_2 = get_rnge_std_hist(ref_feat, query_length, make_norm_simple_subcomponents)
        q_1, q_2 = get_rnge_std_hist(qry_feat, query_length, make_norm_simple_subcomponents)
        d_1, d_2 = get_rnge_std_hist(delta_feat, query_length, make_norm_simple_subcomponents)
        s_3, r_3, q_3, d_3 = [make_norm_simple_subcomponents(arr) for arr in four_group]
        comps  = [s_1, s_2, s_3, r_1, r_2, r_3, q_1, q_2, q_3, d_1, d_2, d_3]
    elif mode == GenMode.LONG_HIST4_NORM_SIMPLE_COMPONENTS.name:
        s_1, s_2 = get_rnge_std_hist(dist_mat, query_length, make_norm_simple_subcomponents)
        r_1, r_2 = get_rnge_std_hist(ref_feat, query_length, make_norm_simple_subcomponents)
        q_1, q_2 = get_rnge_std_hist(qry_feat, query_length, make_norm_simple_subcomponents)
        d_1, d_2 = get_rnge_std_hist(delta_feat, query_length, make_norm_simple_subcomponents)
        s_3, r_3, q_3, d_3 = [make_norm_simple_subcomponents(arr) for arr in four_group]
        comps  = [s_1-s_3, s_2-s_3, r_1-r_3, r_2-r_3, q_1-q_3, q_2-q_3, d_1-d_3, d_2-d_3]
    elif mode == GenMode.LONG_HIST5_NORM_SIMPLE_COMPONENTS.name:
        s_1 = get_mean_hist(dist_mat, query_length, make_norm_simple_subcomponents)
        r_1 = get_mean_hist(ref_feat, query_length, make_norm_simple_subcomponents)
        q_1 = get_mean_hist(qry_feat, query_length, make_norm_simple_subcomponents)
        d_1 = get_mean_hist(delta_feat, query_length, make_norm_simple_subcomponents)
        s_2, r_2, q_2, d_2 = [make_norm_simple_subcomponents(arr) for arr in four_group]
        comps  = [s_1, s_2, r_1, r_2, q_1, q_2, d_1, d_2]
    elif mode == GenMode.LONG_HIST6_NORM_SIMPLE_COMPONENTS.name:
        s_1 = get_mean_hist(dist_mat, query_length, make_norm_simple_subcomponents)
        r_1 = get_mean_hist(ref_feat, query_length, make_norm_simple_subcomponents)
        q_1 = get_mean_hist(qry_feat, query_length, make_norm_simple_subcomponents)
        d_1 = get_mean_hist(delta_feat, query_length, make_norm_simple_subcomponents)
        s_2, r_2, q_2, d_2 = [make_norm_simple_subcomponents(arr) for arr in four_group]
        comps  = [s_1-s_2, r_1-r_2, q_1-q_2, d_1-d_2]
    elif mode == GenMode.MATCH_INDS.name:
        i_1 = np.stack([get_hist(inds, i, query_length) for i in range(len(inds))], axis=1)
        comps  = [i_1]
    elif mode == GenMode.TEST1.name:
        i_1 = np.stack([get_hist(inds, i, query_length) for i in range(len(inds))], axis=1)
        comps  = [make_simple_subcomponents(i) for i in [delta_feat]] + [i_1]
    elif mode == GenMode.TEST2.name:
        i_1 = np.stack([get_hist(inds, i, query_length) for i in range(len(inds))], axis=1)
        comps  = [make_simple_subcomponents(i) for i in four_group] + [i_1]
    else:
        raise BadGenMode(f'Unknown GenMode: {str(mode)}')
    #
    components = np.concatenate(comps, axis=0)
    components[np.where(np.isfinite(components) is False)] = 0
    return np.transpose(components)
