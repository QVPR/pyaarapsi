#!/usr/bin/env python3
'''
Helper commands for VPR
'''
import copy
from enum import Enum, unique
import json
import cv2
from typing import Union, List, Optional, Tuple

from tqdm.auto import tqdm
import numpy as np
from numpy.typing import NDArray
# from numpy.typing import NDArray
from cv2 import resize as cv_resize, cvtColor as cv_cvtColor, INTER_AREA as cv_INTER_AREA, \
    COLOR_RGB2GRAY as cv_COLOR_RGB2GRAY # pylint: disable=E0611

from pyaarapsi.vpr_classes.netvlad import NetVLAD_Container
from pyaarapsi.vpr_classes.hybridnet import HybridNet_Container
from pyaarapsi.vpr_classes.salad import SALAD_Container
from pyaarapsi.vpr_classes.apgem import APGEM_Container
from pyaarapsi.core.helper_tools import perforate, formatException, m2m_dist
from pyaarapsi.pathing.basic import calc_path_stats

# For image processing type
@unique
class FeatureType(Enum):
    '''
    VPR Descriptors
    '''
    RAW                 = 1
    PATCHNORM           = 2
    NETVLAD             = 3
    HYBRIDNET           = 4
    ROLLNORM            = 5
    NORM                = 6
    SALAD               = 7
    APGEM               = 8

    class Exception(Exception):
        '''
        Bad usage.
        '''

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

def getFeatureLength(feature_type: FeatureType, img_dims: list):
    '''
    Get length of vector per FeatureType when used to generate features
    Although measurable after-the-fact, useful for allocating ahead of time.
    '''
    if feature_type.name in [FeatureType.SALAD.name]:
        return 8192
    elif feature_type.name in [FeatureType.NETVLAD.name, FeatureType.APGEM.name,
                               FeatureType.HYBRIDNET.name]:
        return 4096
    elif feature_type.name in [FeatureType.RAW.name, FeatureType.PATCHNORM.name,
                               FeatureType.NORM.name, FeatureType.ROLLNORM.name]:
        return img_dims[0] * img_dims[1]
    else:
        raise FeatureType.Exception(f"Unknown feature type ({str(feature_type)}).")

def overridesImgDims(feature_type: FeatureType):
    '''
    Whether the img_dims parameter is used for each FeatureType.
    '''
    if feature_type.name in [FeatureType.RAW.name, FeatureType.PATCHNORM.name,
                             FeatureType.ROLLNORM.name, FeatureType.NORM.name]:
        return True
    elif feature_type.name in [FeatureType.NETVLAD.name, FeatureType.HYBRIDNET.name,
                               FeatureType.SALAD.name, FeatureType.APGEM.name]:
        return False
    else:
        raise Exception("Unknown feature type (%s)." % str(feature_type))

def isFeatureSpatiallyRelated(feature_type: FeatureType):
    '''
    Whether elements of the feature vector relate.
    '''
    if feature_type.name in [FeatureType.RAW.name, FeatureType.PATCHNORM.name,
                             FeatureType.ROLLNORM.name, FeatureType.NORM.name,
                             FeatureType.HYBRIDNET.name]:
        return True
    elif feature_type.name in [FeatureType.NETVLAD.name, FeatureType.SALAD.name,
                               FeatureType.APGEM.name]:
        return False
    else:
        raise Exception("Unknown feature type (%s)." % str(feature_type))

class ViewMode(Enum):
    '''
    Unused.
    '''
    FORWARD  	        = 0
    FORWARDRIGHT 	    = 1
    RIGHT 		        = 2
    BACKRIGHT 	        = 3
    BACK 		        = 4
    BACKLEFT 	        = 5
    LEFT 		        = 6
    FORWARDLEFT 	    = 7
    PANORAMA 	        = 8
    forward             = 9

class VPR_Tolerance_Mode(Enum):
    '''
    How to generate gt distances and errors
    '''
    METRE_CROW_TRUE     = 0
    METRE_CROW_MATCH    = 1
    METRE_LINE          = 2
    FRAME               = 3

class SVM_Tolerance_Mode(Enum):
    '''
    How to determine whether a match is in-tolerance
    '''
    DISTANCE            = 0
    FRAME               = 1
    TRACK_DISTANCE      = 2

def make_dataset_dictionary(bag_name: str,
                            npz_dbp: str = "/data/compressed_sets",
                            bag_dbp: str = "/data/rosbags",
                            odom_topic: Union[str, List[str]] = "/odom/true",
                            img_topics: List[str] = ["/ros_indigosdk_occam/image0/compressed"],
                            sample_rate: Union[int, float] = 5.0,
                            ft_types: List[str] = [FeatureType.RAW.name],
                            img_dims: List[int] = [64,64],
                            filters: Union[str, dict] = {}):

    '''
    Function to help remember the contents of a VPRDatasetProcessor dataset_params dictionary
    '''
    return dict(bag_name=bag_name, npz_dbp=npz_dbp, bag_dbp=bag_dbp, odom_topic=odom_topic,
                img_topics=img_topics, sample_rate=sample_rate, ft_types=ft_types,
                img_dims=img_dims, filters=filters)

def make_svm_dictionary(ref: dict, qry: dict, factors: List[str], tol_thres: float = 0.5,
                        tol_mode: SVM_Tolerance_Mode = SVM_Tolerance_Mode.DISTANCE,
                        svm_dbp='/cfg/svm_models'):
    '''
    Function to help remember the contents of an SVMModelProcessor params dictionary
    '''
    assert ref['npz_dbp'] == qry['npz_dbp']
    assert ref['bag_dbp'] == qry['bag_dbp']
    svm_svm_dict = dict(factors=factors, tol_thres=tol_thres, tol_mode=tol_mode)
    return  dict(ref=ref, qry=qry, svm=svm_svm_dict, npz_dbp=ref['npz_dbp'],
                 bag_dbp=ref['bag_dbp'], svm_dbp=svm_dbp)

def correct_filters(_filters: Union[dict, str]) -> Tuple[dict, bool]:
    _filters_out = copy.deepcopy(_filters)
    fixed = False
    if isinstance(_filters_out, str):
        if not len(_filters_out): 
            return {}, True
        else:
            _filters_out = json.loads(_filters_out.replace("'", '"'))
        fixed = True
    if 'distance' in _filters_out:
        if _filters_out['distance'] == 0: 
            _filters_out.pop('distance')
            fixed = True
    if 'perforate' in _filters_out:
        randomness  = _filters_out['perforate']['randomness']  if 'randomness'  in _filters_out['perforate'] else 0
        num_holes   = _filters_out['perforate']['num_holes']   if 'num_holes'   in _filters_out['perforate'] else 0
        hole_damage = _filters_out['perforate']['hole_damage'] if 'hole_damage' in _filters_out['perforate'] else 0
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
        if not len(_filters_out['delete-segments']): _filters_out.pop('delete-segments')
    return _filters_out, fixed

def filter_dataset(dataset_in, _filters: Optional[dict] = None, _printer=lambda *args, **kwargs: None):
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
                _printer('[filter_dataset] Filters: %s' % str(_filters))
            except:
                raise Exception('[filter_dataset] Failed to load filter parameters, string: <%s>. Code: %s' % (_filt_str, formatException()))
        else:
            raise Exception("[filter_dataset] Failed to load filter parameters. Got type %s, expected type <class 'str'> or <class 'dict'>." % (str(type(_filters))))
    try:
        if 'distance' in _filters.keys():
            _printer('[filter_dataset] Filtering by distance: %s' % str(_filters['distance']))
            distance_threshold      = _filters['distance']
            try:
                xy = np.stack([dataset_in['dataset']['px'][:,0], dataset_in['dataset']['py'][:,0]], axis=1)
                _printer('[filter_dataset] Using first column of px, py for xy array to perform distance filtering.')
            except IndexError:
                xy = np.stack([dataset_in['dataset']['px'], dataset_in['dataset']['py']], axis=1)
            if not ((xy.shape[1] == 2) and (xy.shape[0] > 1)):
                raise Exception('Could not build xy array with two columns with more than one entry in each; please check odometry.')
            xy_sum, xy_len          = calc_path_stats(xy)
            filt_indices            = [np.argmin(np.abs(xy_sum-(distance_threshold*i))) for i in np.arange(int((1/distance_threshold) * xy_len))]
            dataset_out             = {'params': dataset_in['params']}
            dataset_out['dataset']  = {key: dataset_in['dataset'][key][filt_indices] for key in dataset_in['dataset'].keys()}
            
            _filters.pop('distance')
            return filter_dataset(dataset_out, _filters=_filters)
        elif 'perforate' in _filters.keys():
            _printer('[filter_dataset] Filtering by perforation: %s' % str(_filters['perforate']))
            randomness  = _filters['perforate']['randomness']   if 'randomness'     in _filters['perforate']    else None
            num_holes   = _filters['perforate']['num_holes']    if 'num_holes'      in _filters['perforate']    else None
            hole_damage = _filters['perforate']['hole_damage']  if 'hole_damage'    in _filters['perforate']    else None
            offset      = _filters['perforate']['offset']       if 'offset'         in _filters['perforate']    else None
            _override   = _filters['perforate']['_override']    if '_override'      in _filters['perforate']    else 0
            filt_indices = perforate(_len=dataset_in['dataset']['time'].shape[0], randomness=randomness, \
                                     num_holes=num_holes, hole_damage=hole_damage, offset=offset, _override=_override)
            dataset_out             = {'params': dataset_in['params']}
            dataset_out['dataset']  = {key: dataset_in['dataset'][key][filt_indices] for key in dataset_in['dataset'].keys()}
            
            _filters.pop('perforate')
            return filter_dataset(dataset_out, _filters=_filters)
        elif 'forward-only' in _filters.keys():
            if _filters['forward-only']:
                try:
                    filt_indices        = [True if i >= 0 else False for i in dataset_in['dataset']['vx'][:,0]]
                    _printer('[filter_dataset] Using first column of vx array to perform direction filtering.')
                except IndexError:
                    filt_indices        = [True if i >= 0 else False for i in dataset_in['dataset']['vx']]
                dataset_out             = {'params': dataset_in['params']}
                dataset_out['dataset']  = {key: dataset_in['dataset'][key][filt_indices] for key in dataset_in['dataset'].keys()}
            else:
                dataset_out             = dataset_in    
            _filters.pop('forward-only')
            return filter_dataset(dataset_out, _filters=_filters)
        elif 'crop-loop' in _filters.keys():
            if _filters['crop-loop']:
                try:
                    xy = np.stack([dataset_in['dataset']['px'][:,0], dataset_in['dataset']['py'][:,0]], axis=1)
                    _printer('[filter_dataset] Using first column of px, py for xy array to perform distance filtering.')
                except IndexError:
                    xy = np.stack([dataset_in['dataset']['px'], dataset_in['dataset']['py']], axis=1)
                xy_start = copy.deepcopy(xy[0:1,:])
                xy[0:int(xy.shape[0] / 2),:] = 1000
                overlap_ind = np.argmin(m2m_dist(xy_start, xy))
                dataset_out             = {'params': dataset_in['params']}
                dataset_out['dataset']  = {key: dataset_in['dataset'][key][:overlap_ind] for key in dataset_in['dataset'].keys()}
            else:
                dataset_out = dataset_in
            _filters.pop('crop-loop')
            return filter_dataset(dataset_out, _filters=_filters)
        elif 'crop-bounds' in _filters.keys():
            dataset_out             = {'params': dataset_in['params']}
            dataset_out['dataset']  = {key: dataset_in['dataset'][key][slice(*_filters['crop-bounds'])] for key in dataset_in['dataset'].keys()}
            _filters.pop('crop-bounds')
            return filter_dataset(dataset_out, _filters=_filters)
        elif 'delete-segments' in _filters.keys():
            dataset_out             = {'params': dataset_in['params']}
            _preserved              = np.ones(len(dataset_in['dataset']['px']), dtype=bool)
            for _start, _end in _filters['delete-segments']:
                _preserved[_start:_end] = False
            dataset_out['dataset']  = {key: dataset_in['dataset'][key][_preserved] for key in dataset_in['dataset'].keys()}
            _filters.pop('delete-segments')
            return filter_dataset(dataset_out, _filters=_filters)
        else:
            return dataset_in
    except:
        raise Exception('[filter_dataset] Failed. Code: %s' % formatException())

def discretise(dict_in, metrics=None, mode=None, keep='first'):
    if not len(dict_in):
        raise Exception("[filter] Full dictionary not yet built.")
    filtered = copy.deepcopy(dict_in) # ensure we don't change the original dictionary

    if mode is None:
        return filtered
    valid_keeps = ['first', 'random', 'average']
    if not keep in valid_keeps: # ensure valid
        raise Exception('[filter] Unsupported keep style %s. Valid keep styles: %s' % (str(keep), str(valid_keeps)))
    valid_modes = ['position', 'velocity']
    if not mode in valid_modes: # ensure valid
        raise Exception('[filter] Unsupported mode %s. Valid modes: %s' % (str(mode), str(valid_modes)))
    
    # Perform filter step:
    if mode in ['position', 'velocity']: # valid inputs to roundSpatial()
        (filtered['odom'][mode], groupings) = roundSpatial(filtered['odom'][mode], metrics)
    #elif mode in []: #TODO
    #    pass
    else:
        return None
    
    filtered = keep_operation(filtered, groupings, keep) # remove duplications
    return filtered

def roundSpatial(spatial_vec, metrics=None):
    if metrics is None:
        metrics = {'x': 0.05, 'y': 0.05, 'yaw': (2*np.pi/360)}
    new_spatial_vec = {}
    for key in metrics:
        new_spatial_vec[key] = np.round(np.array(spatial_vec[key])/metrics[key],0) * metrics[key]
    new_spatial_matrix = np.transpose(np.stack([new_spatial_vec[key] for key in list(new_spatial_vec)]))
    groupings = []
    for arr in np.unique(new_spatial_matrix, axis=0): # for each unique row combination:
        groupings.append(list(np.array(np.where(np.all(new_spatial_matrix==arr,axis=1))).flatten())) # indices
        #groupings.append(list(np.array((np.all(new_spatial_matrix==arr,axis=1))).flatten())) # bools
        return new_spatial_vec, groupings
    
def keep_operation(d_in, groupings, mode='first'):
# Note: order of groupings within the original set can be lost depending on how groupings were generated
    
    # create 'table' where all rows are the same length with first entry as the 'label' (tuple)
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
        raise Exception("Fatal")
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

def getFeat(im: Union[np.ndarray, List[np.ndarray]], fttypes: Union[FeatureType, List[FeatureType]],
            dims: list, use_tqdm: bool = False,
            nn_hybrid: Optional[HybridNet_Container] = None,
            nn_netvlad: Optional[NetVLAD_Container] = None,
            nn_salad: Optional[SALAD_Container] = None,
            nn_apgem: Optional[APGEM_Container] = None) -> Union[np.ndarray, List[np.ndarray]]:
    
    ft_list     = []
    if not isinstance(im, list):
        _im = [im]
    else:
        _im = im
    if not isinstance(fttypes, list):
        _fttypes = [fttypes]
    else:
        _fttypes = fttypes
    for fttype in _fttypes:
        if fttype.name in [FeatureType.RAW.name, FeatureType.PATCHNORM.name, FeatureType.ROLLNORM.name, FeatureType.NORM.name]:
            ft_ready_list = []
            if use_tqdm: 
                iter_obj = tqdm(_im)
            else: 
                iter_obj = _im
            for i in iter_obj:
                imr = cv_resize(i, dims)
                ft  = cv_cvtColor(imr, cv_COLOR_RGB2GRAY)
                if fttype.name == FeatureType.PATCHNORM.name:
                    ft = patchNormaliseImage(ft, 8)
                elif fttype.name == FeatureType.ROLLNORM.name:
                    ft = rollNormaliseImage(ft, 2)
                elif fttype.name == FeatureType.NORM.name:
                    ft = normaliseImage(ft)
                ft_ready_list.append(ft.flatten())
            if len(ft_ready_list) == 1:
                ft_ready = ft_ready_list[0]
            else:
                ft_ready = np.stack(ft_ready_list)
        elif fttype.name == FeatureType.HYBRIDNET.name and not nn_hybrid is None:
            ft_ready = nn_hybrid.getFeat(_im, use_tqdm=use_tqdm)
        elif fttype.name == FeatureType.NETVLAD.name and not nn_netvlad is None:
            ft_ready = nn_netvlad.getFeat(_im, use_tqdm=use_tqdm)
        elif fttype.name == FeatureType.SALAD.name and not nn_salad is None:
            ft_ready = nn_salad.getFeat(_im, use_tqdm=use_tqdm)
        elif fttype.name == FeatureType.APGEM.name and not nn_apgem is None:
            ft_ready = nn_apgem.getFeat(_im, use_tqdm=use_tqdm)
        else:
            raise FeatureType.Exception("[getFeat] fttype could not be handled.")
        ft_list.append(ft_ready)
    if len(ft_list) == 1: 
        return ft_list[0]
    return ft_list

def patchNormaliseImage(img, patchLength):
# take input image, divide into regions, normalise
# returns: patch normalised image
# TODO: vectorize
    img1 = img.astype(float)
    img2 = img1.copy()
    if patchLength == 1: # single pixel; already p-n'd
        return img2
    for i in range(img1.shape[0]//patchLength): # floor division -> number of rows
        iStart = i*patchLength
        iEnd = (i+1)*patchLength
        for j in range(img1.shape[1]//patchLength): # floor division -> number of cols
            jStart = j*patchLength
            jEnd = (j+1)*patchLength
            mean1 = np.mean(img1[iStart:iEnd, jStart:jEnd])
            std1 = np.std(img1[iStart:iEnd, jStart:jEnd])
            # offset remove mean:
            img2[iStart:iEnd, jStart:jEnd] = img1[iStart:iEnd, jStart:jEnd] - mean1
            if std1 == 0:
                std1 = 0.1
            img2[iStart:iEnd, jStart:jEnd] /= std1 # crush by std
    return img2

def normaliseImage(img):
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

def rollNormaliseImage(img, kernel_size):
    '''
    take input image and use a rolling kernel to noramlise
    returns: rolling-kernel-normalised image
    TODO: pad with average value of image to fix edge artefacts
    TODO: reduce square artefacts by transitioning to circular average region
    '''
    img1            = img.astype(float)
    img2            = img1.copy()
    if kernel_size == 1: # single pixel; already p-n'd
        return img2
    k_options       = list(range(-kernel_size,kernel_size+1,1))
    rolled_stack    = np.dstack([np.roll(np.roll(img2,i,0),j,1)
                                 for j in k_options for i in k_options])
    #rollnormed      = 255 - (np.mean(rolled_stack, 2) / np.std(rolled_stack, 2))
    rollnormed      = np.mean(rolled_stack, 2)
    return rollnormed
