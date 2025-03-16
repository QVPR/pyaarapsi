#!/usr/bin/env python3
'''
Helper commands for VPR
'''
from __future__ import annotations
import copy
from enum import Enum
import json
from typing import Union, List, Optional, Tuple, Callable

from tqdm.auto import tqdm
import numpy as np
from numpy.typing import NDArray
from cv2 import resize as cv_resize, cvtColor as cv_cvtColor, INTER_AREA as cv_INTER_AREA, \
    COLOR_RGB2GRAY as cv_COLOR_RGB2GRAY # pylint: disable=E0611

from pyaarapsi.vpr.classes.descriptors.generic import DescriptorContainer
from pyaarapsi.vpr.classes.vprdescriptor import VPRDescriptor
from pyaarapsi.vpr.classes.dimensions import ImageDimensions
from pyaarapsi.vpr.classes.data.rosbagdataset import RosbagDataset
from pyaarapsi.core.helper_tools import perforate, format_exception, m2m_dist
from pyaarapsi.pathing.basic import calc_path_stats

def make_features_presentable(features: NDArray[np.float64],
                              out_dims: Optional[Tuple[int,int]] = None,
                              feature_dims: Optional[Tuple[int,int]] = None) -> NDArray[np.float64]:
    '''
    This function will reshape an input feature vector into a 2D, presentable image ready for
    matplotlib. The image will have [width, height] == out_dims.
    If the feature dimensions are not provided, the vector will be reshaped into a rectangle with
    equivalent/similar aspect ratio (padding additional pixels with zero).
    If the feature dimensions are provided, the vector will be reshaped directly into these.
    The output image is generated using:
    >>> cv2.resize(..., (out_dims[1], out_dims[0]), interpolation=cv2.INTER_AREA)
    This ensures np.reshape and cv2.resize use the same ordering of dimensions.
    '''
    if out_dims is None:
        out_dims = [64,64]
    assert np.array(out_dims).shape == (2,)
    features_copy = copy.deepcopy(features)
    if not feature_dims is None:
        out_arr = np.reshape(features_copy, feature_dims).astype(float)
        return cv_resize(out_arr, (out_dims[1], out_dims[0]), interpolation=cv_INTER_AREA)
    features_flat = features_copy.flatten()
    new_h = int(np.round(np.sqrt(features_flat.shape[0] * out_dims[0] / out_dims[1])))
    new_w = int(np.round(features_flat.shape[0] / new_h))
    while (new_h * new_w) < features_flat.shape[0]:
        if new_h < new_w:
            new_w += 1
        else:
            new_h += 1
    out_arr = np.zeros((new_h, new_w)).astype(float)
    out_arr_view = out_arr.view().reshape(-1)
    out_arr_view[:features_flat.shape[0]] = features_flat
    return cv_resize(out_arr, (out_dims[1], out_dims[0]), interpolation=cv_INTER_AREA)

class VPRToleranceMode(Enum):
    '''
    How to generate gt distances and errors
    '''
    METRE_CROW_TRUE     = 0
    METRE_CROW_MATCH    = 1
    METRE_LINE          = 2
    FRAME               = 3

def get_top_match_inds(ref_feats: NDArray, qry_feats: NDArray, top_k: int = 20):
    '''
    Get top k match indices
    '''
    sim_matrix = m2m_dist(ref_feats, qry_feats)
    return np.argsort(sim_matrix, axis=0)[:top_k]

def correct_filters(_filters: Union[dict, str]) -> Tuple[dict, bool]:
    '''
    Convert a filter from json string to dictionary
    '''
    _filters_out = copy.deepcopy(_filters)
    fixed = False
    if isinstance(_filters_out, str):
        if _filters_out in ["", "{}"]:
            return {}, True
        else:
            _filters_out = json.loads(_filters_out.replace("'", '"'))
        fixed = True
    if 'distance' in _filters_out:
        if _filters_out['distance'] == 0:
            _filters_out.pop('distance')
            fixed = True
    if 'perforate' in _filters_out:
        randomness  = _filters_out['perforate']['randomness']  \
                            if 'randomness'  in _filters_out['perforate'] else 0
        num_holes   = _filters_out['perforate']['num_holes']   \
                            if 'num_holes'   in _filters_out['perforate'] else 0
        hole_damage = _filters_out['perforate']['hole_damage'] \
                            if 'hole_damage' in _filters_out['perforate'] else 0
        if (randomness == 0) and ((num_holes == 0) or (hole_damage == 0)):
            _filters_out.pop('perforate')
            fixed = True
    if 'forward-only' in _filters_out:
        if not _filters_out['forward-only']:
            _filters_out.pop('forward-only')
            fixed = True
    if 'crop-loop' in _filters_out:
        if not _filters_out['crop-loop']:
            _filters_out.pop('crop-loop')
            fixed = True
    if 'crop-bounds' in _filters_out:
        if all(i is None for i in _filters_out['crop-bounds']):
            _filters_out.pop('crop-bounds')
            fixed = True
        elif _filters_out['crop-bounds'][0] == 0:
            _filters_out.pop('crop-bounds')
            fixed = True
    if 'delete-segments' in _filters_out:
        if len(_filters_out['delete-segments']) == 0:
            _filters_out.pop('delete-segments')
    return _filters_out, fixed

def filter_image_dataset(dataset: RosbagDataset, \
    _printer: Callable = lambda *args, **kwargs: None) -> RosbagDataset:
    '''
    Apply image space filters
    '''
    data_out = copy.deepcopy(dataset.data)
    for image_filter in dataset.params.image_filters:
        data_out = image_filter.apply(data=data_out, params=dataset.params)
    return dataset.populate(params=dataset.params, data=data_out)

def filter_feature_dataset(dataset: RosbagDataset, \
    _printer: Callable = lambda *args, **kwargs: None) -> RosbagDataset:
    '''
    Apply feature space filters
    '''
    data_out = copy.deepcopy(dataset.data)
    for feature_filter in dataset.params.feature_filters:
        data_out = feature_filter.apply(data=data_out, params=dataset.params)
    return dataset.populate(params=dataset.params, data=data_out)

def filter_dataset(dataset_in, _filters: Optional[dict] = None, _printer=lambda *args,
                   **kwargs: None):
    '''
    Legacy. Super method for filtering a dataset dictionary
    '''
    if _filters is None:
        if isinstance(dataset_in['params']['filters'], dict):
            _filters = copy.deepcopy(dataset_in['params']['filters'])
        elif isinstance(dataset_in['params']['filters'], str):
            _filt_str = str(dataset_in['params']['filters']).replace("'", '"')
            try:
                if not _filt_str:
                    return dataset_in
                _filters = json.loads(_filt_str)
                if not _filters:
                    return dataset_in
                _printer(f'[filter_dataset] Filters: {str(_filters)}')
            except Exception as e:
                raise ValueError('[filter_dataset] Failed to load filter parameters, string: '
                                f'<{_filt_str}>. Code: {format_exception()}') from e
        else:
            raise ValueError("[filter_dataset] Failed to load filter parameters. Got type "
                             f"{str(type(_filters))}, expected type <class 'str'> or "
                             "<class 'dict'>.")
    try:
        if 'distance' in _filters.keys():
            _printer(f"[filter_dataset] Filtering by distance: {str(_filters['distance'])}")
            distance_threshold      = _filters['distance']
            try:
                xy = np.stack([dataset_in['dataset']['px'][:,0],
                               dataset_in['dataset']['py'][:,0]], axis=1)
                _printer('[filter_dataset] Using first column of px, py for xy array to perform '
                         'distance filtering.')
            except IndexError:
                xy = np.stack([dataset_in['dataset']['px'], dataset_in['dataset']['py']], axis=1)
            if not ((xy.shape[1] == 2) and (xy.shape[0] > 1)):
                raise ValueError('Could not build xy array with two columns with more than one '
                                 'entry in each; please check odometry.')
            xy_sum, xy_len          = calc_path_stats(xy)
            filt_indices            = [np.argmin(np.abs(xy_sum-(distance_threshold*i))) \
                                        for i in np.arange(int((1/distance_threshold) * xy_len))]
            dataset_out             = {'params': dataset_in['params']}
            dataset_out['dataset']  = {key: dataset_in['dataset'][key][filt_indices] \
                                        for key in dataset_in['dataset'].keys()}
            _filters.pop('distance')
            return filter_dataset(dataset_out, _filters=_filters)
        elif 'perforate' in _filters.keys():
            perf_filts  = _filters['perforate']
            _printer(f"[filter_dataset] Filtering by perforation: {str(perf_filts)}")
            randomness  = perf_filts['randomness']  if 'randomness'  in perf_filts else None
            num_holes   = perf_filts['num_holes']   if 'num_holes'   in perf_filts else None
            hole_damage = perf_filts['hole_damage'] if 'hole_damage' in perf_filts else None
            offset      = perf_filts['offset']      if 'offset'      in perf_filts else None
            _override   = perf_filts['_override']   if '_override'   in perf_filts else 0
            filt_indices = perforate(_len=dataset_in['dataset']['time'].shape[0], \
                            randomness=randomness, num_holes=num_holes, hole_damage=hole_damage, \
                                offset=offset, _override=_override)
            dataset_out             = {'params': dataset_in['params']}
            dataset_out['dataset']  = {key: dataset_in['dataset'][key][filt_indices] \
                                       for key in dataset_in['dataset'].keys()}
            _filters.pop('perforate')
            return filter_dataset(dataset_out, _filters=_filters)
        elif 'forward-only' in _filters.keys():
            if _filters['forward-only']:
                try:
                    filt_indices        = [True if i >= 0 else False \
                                           for i in dataset_in['dataset']['vx'][:,0]]
                    _printer('[filter_dataset] Using first column of vx array to perform '
                             'direction filtering.')
                except IndexError:
                    filt_indices        = [True if i >= 0 else False \
                                           for i in dataset_in['dataset']['vx']]
                dataset_out             = {'params': dataset_in['params']}
                dataset_out['dataset']  = {key: dataset_in['dataset'][key][filt_indices] \
                                           for key in dataset_in['dataset'].keys()}
            else:
                dataset_out             = dataset_in
            _filters.pop('forward-only')
            return filter_dataset(dataset_out, _filters=_filters)
        elif 'crop-loop' in _filters.keys():
            if _filters['crop-loop']:
                try:
                    xy = np.stack([dataset_in['dataset']['px'][:,0],
                                   dataset_in['dataset']['py'][:,0]], axis=1)
                    _printer('[filter_dataset] Using first column of px, py for xy array to '
                             'perform distance filtering.')
                except IndexError:
                    xy = np.stack([dataset_in['dataset']['px'],
                                   dataset_in['dataset']['py']], axis=1)
                xy_start = copy.deepcopy(xy[0:1,:])
                xy[0:int(xy.shape[0] / 2),:] = 1000
                overlap_ind = np.argmin(m2m_dist(xy_start, xy))
                dataset_out             = {'params': dataset_in['params']}
                dataset_out['dataset']  = {key: dataset_in['dataset'][key][:overlap_ind] \
                                           for key in dataset_in['dataset'].keys()}
            else:
                dataset_out = dataset_in
            _filters.pop('crop-loop')
            return filter_dataset(dataset_out, _filters=_filters)
        elif 'crop-bounds' in _filters.keys():
            dataset_out             = {'params': dataset_in['params']}
            dataset_out['dataset']  = {key: \
                                       dataset_in['dataset'][key][slice(*_filters['crop-bounds'])] \
                                        for key in dataset_in['dataset'].keys()}
            _filters.pop('crop-bounds')
            return filter_dataset(dataset_out, _filters=_filters)
        elif 'delete-segments' in _filters.keys():
            dataset_out             = {'params': dataset_in['params']}
            _preserved              = np.ones(len(dataset_in['dataset']['px']), dtype=bool)
            for _start, _end in _filters['delete-segments']:
                _preserved[_start:_end] = False
            dataset_out['dataset']  = {key: dataset_in['dataset'][key][_preserved] \
                                       for key in dataset_in['dataset'].keys()}
            _filters.pop('delete-segments')
            return filter_dataset(dataset_out, _filters=_filters)
        else:
            return dataset_in
    except Exception as e:
        raise ValueError(f'[filter_dataset] Failed. Code: {format_exception()}') from e

def discretise(dict_in, metrics=None, mode=None, keep='first'):
    '''
    Legacy. Performs a discretisation operation on a dataset dictionary.
    '''
    if len(dict_in) == 0:
        raise ValueError("[filter] Full dictionary not yet built.")
    filtered = copy.deepcopy(dict_in) # ensure we don't change the original dictionary

    if mode is None:
        return filtered
    valid_keeps = ['first', 'random', 'average']
    if keep not in valid_keeps: # ensure valid
        raise ValueError(f'[filter] Unsupported keep style {str(keep)}. '
                        f'Valid keep styles: {str(valid_keeps)}')
    valid_modes = ['position', 'velocity']
    if mode not in valid_modes: # ensure valid
        raise ValueError(f'[filter] Unsupported mode {str(mode)}. '
                        f'Valid modes: {str(valid_modes)}')
    # Perform filter step:
    if mode in ['position', 'velocity']: # valid inputs to roundSpatial()
        (filtered['odom'][mode], groupings) = round_spatial(filtered['odom'][mode], metrics)
    else:
        return None
    filtered = keep_operation(filtered, groupings, keep) # remove duplications
    return filtered

def round_spatial(spatial_vec, metrics=None):
    '''
    Legacy. Rounds components to some increment
    '''
    if metrics is None:
        metrics = {'x': 0.05, 'y': 0.05, 'yaw': (2*np.pi/360)}
    new_spatial_vec = {}
    for key in metrics:
        new_spatial_vec[key] = np.round(np.array(spatial_vec[key])/metrics[key],0) * metrics[key]
    new_spatial_matrix = np.transpose(np.stack([new_spatial_vec[key] \
                                                for key in list(new_spatial_vec)]))
    groupings = []
    for arr in np.unique(new_spatial_matrix, axis=0): # for each unique row combination:
        groupings.append(np.array(np.where(np.all(new_spatial_matrix==arr,axis=1))\
                                  ).flatten().tolist()) # indices
        # groupings.append(np.array(np.all(new_spatial_matrix==arr,axis=1)\
        #                           ).flatten().tolist()) # bools
        return new_spatial_vec, groupings

def keep_operation(d_in, groupings, mode='first'):
    '''
    Legacy. Performs a grouping depending on a mode, resulting in a reduced set size.
    '''
    # Note: order of groupings within the original set can be lost depending on how groupings
    # were generated create 'table' where all rows are the same length with first entry as
    # the 'label' (tuple)
    dict_to_list = []
    for bigkey in ['odom', 'img_feats']:
        for midkey in set(d_in[bigkey].keys()):
            for lowkey in set(d_in[bigkey][midkey].keys()):
                base = [(bigkey, midkey, lowkey)]
                base.extend(d_in[bigkey][midkey][lowkey])
                dict_to_list.append(base)
    times = [('times',)]
    times.extend(d_in['times'])
    dict_to_list.append(times)
    np_dict_to_list = np.transpose(np.array(dict_to_list, dtype=object))

    # extract rows
    groups_store = []
    for group in groupings:
        if len(group) < 1:
            continue
        if mode=='average':
            groups_store.append(np.mean(np_dict_to_list[1:, :][group,:], axis=0))
            continue
        elif mode=='first':
            index = 0
        elif mode=='random':
            index = int(np.random.rand() * (len(group) - 1))
        else:
            continue
        # for first and random modes:
        index_to_keep = group[index] + 1 # +1 accounts for label
        groups_store.append(np_dict_to_list[index_to_keep, :])

    # restructure and reorder
    cropped_store = np.array(groups_store)
    ind = -1
    for c, i in enumerate(np_dict_to_list[0,:]):
        if i[0] == 'times':
            ind = c
            break
    if ind == -1:
        raise ValueError("Fatal")
    cropped_reorder = cropped_store[cropped_store[:,-1].argsort()]
    d_in.pop('times')
    d_in['times'] = cropped_reorder[:,ind]

    # convert back to dictionary and update old dictionary entries
    for bigkey in ['odom', 'img_feats']:
        for midkey in set(d_in[bigkey].keys()):
            for lowkey in set(d_in[bigkey][midkey].keys()):
                for c, i in enumerate(np_dict_to_list[0,:]):
                    if (bigkey,midkey,lowkey) == i:
                        d_in[bigkey][midkey].pop(lowkey)
                        d_in[bigkey][midkey][i[2]] = np.stack(cropped_reorder[:,c],axis=0)
    return d_in

def get_feat(   im: Union[np.ndarray, List[np.ndarray]],
                descriptors: Union[VPRDescriptor, List[VPRDescriptor]],
                dims: Optional[ImageDimensions] = None, use_tqdm: bool = False,
                containers: Optional[dict[str, DescriptorContainer]] = None
                ) -> Union[np.ndarray, List[np.ndarray]]:
    '''
    Feature extraction for VPR, using containers and cv2 operations
    '''
    try:
        features_list     = []
        if isinstance(im, list):
            im_in = im
        else:
            im_in = [im]
        if not isinstance(descriptors, list):
            descriptors_in = [descriptors]
        else:
            descriptors_in = descriptors
        for descriptor in descriptors_in:
            if descriptor in VPRDescriptor.containerless_descriptors():
                generated_features_list = []
                for i in tqdm(im_in) if use_tqdm else im_in:
                    imr = cv_resize(i, dims.for_cv(), interpolation=cv_INTER_AREA)
                    ft  = cv_cvtColor(imr, cv_COLOR_RGB2GRAY)
                    if descriptor == VPRDescriptor.PATCHNORM:
                        ft = patch_normalise_image(ft, 8)
                    elif descriptor == VPRDescriptor.ROLLNORM:
                        ft = roll_normalise_image(ft, 2)
                    elif descriptor == VPRDescriptor.NORM:
                        ft = normalise_image(ft)
                    generated_features_list.append(ft.flatten())
                if len(generated_features_list) == 1:
                    generated_features = generated_features_list[0]
                else:
                    generated_features = np.stack(generated_features_list)
            elif descriptor in VPRDescriptor.descriptors_with_container():
                container = containers[descriptor.name]
                generated_features = container.get_feat(dataset_input=im_in, dims=None,
                                                        use_tqdm=use_tqdm, save_dir=None)
            else:
                raise VPRDescriptor.Exception(f"[get_feat] descriptor {descriptor.name} "
                                              "could not be handled.")
            features_list.append(generated_features)
        if len(features_list) == 1:
            return features_list[0]
        return features_list
    except Exception as e:
        raise VPRDescriptor.Exception("[get_feat] failed to process inputs.") from e

def patch_normalise_image(img: NDArray, patch_length: int):
    """
    # take input image, divide into regions, normalise
    # returns: patch normalised image
    # TODO: vectorize
    """
    assert isinstance(img, np.ndarray) and isinstance(patch_length, int)
    img_edit = copy.deepcopy(img.astype(float))
    if patch_length == 1: # single pixel; already p-n'd
        return img_edit
    for i in range(img_edit.shape[0]//patch_length): # floor division -> number of rows
        i_start = i*patch_length
        i_end = (i+1)*patch_length
        for j in range(img_edit.shape[1]//patch_length): # floor division -> number of cols
            j_start = j*patch_length
            j_end = (j+1)*patch_length
            mean1 = np.mean(img_edit[i_start:i_end, j_start:j_end])
            std1 = np.std(img_edit[i_start:i_end, j_start:j_end])
            # offset remove mean:
            img_edit[i_start:i_end, j_start:j_end] = img[i_start:i_end, j_start:j_end] - mean1
            if std1 == 0:
                std1 = 0.1
            img_edit[i_start:i_end, j_start:j_end] /= std1 # crush by std
    return img_edit

def normalise_image(img):
    '''
    Normalise whole image to be bounded between -1 and 1.
    '''
    img1 = img.astype(float)
    img2 = img1.copy()
    _mean = np.mean(img2.flatten())
    _std  = np.std(img2.flatten())
    if _std == 0:
        _std = 0.1
    return (img - _mean) / _std

def roll_normalise_image(img: NDArray, kernel_size: int):
    '''
    take input image and use a rolling kernel to noramlise
    returns: rolling-kernel-normalised image
    TODO: pad with average value of image to fix edge artefacts
    TODO: reduce square artefacts by transitioning to circular average region
    '''
    assert isinstance(img, np.ndarray) and isinstance(kernel_size, int)
    img_edit = copy.deepcopy(img.astype(float))
    if kernel_size == 1: # single pixel; already roll normalised
        return img_edit
    k_options       = list(range(-kernel_size,kernel_size+1,1))
    rolled_stack    = np.dstack([np.roll(np.roll(img_edit,i,0),j,1)
                                 for j in k_options for i in k_options])
    #rollnormed      = 255 - (np.mean(rolled_stack, 2) / np.std(rolled_stack, 2))
    rollnormed      = np.mean(rolled_stack, 2)
    return rollnormed
