import copy
import numpy as np
import cv2
from enum import Enum
from tqdm.auto import tqdm

# For image processing type
class FeatureType(Enum):
    NONE        = 0
    RAW         = 1
    PATCHNORM   = 2
    NETVLAD     = 3
    HYBRIDNET   = 4
    ROLLNORM    = 5

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
    if ind == -1: raise Exception("Fatal")
    cropped_reorder = cropped_store[cropped_store[:,-1].argsort()]
    d_in.pop('times')
    d_in['times'] = cropped_reorder[:,c]

    # convert back to dictionary and update old dictionary entries
    for bigkey in ['odom', 'img_feats']:
        for midkey in set(d_in[bigkey].keys()):
            for lowkey in set(d_in[bigkey][midkey].keys()):
                for c, i in enumerate(np_dict_to_list[0,:]):
                    if (bigkey,midkey,lowkey) == i:
                        d_in[bigkey][midkey].pop(lowkey)
                        d_in[bigkey][midkey][i[2]] = np.stack(cropped_reorder[:,c],axis=0)
    return d_in

def getFeat(im, fttypes, dims, use_tqdm=False, nn_hybrid=None, nn_netvlad=None):
    ft_list     = []
    req_mode    = isinstance(im, list)

    for fttype in fttypes:
        if fttype in [FeatureType.RAW, FeatureType.PATCHNORM, FeatureType.ROLLNORM]:
            if not req_mode:
                im = [im]
            ft_ready_list = []
            if use_tqdm: iter_obj = tqdm(im)
            else: iter_obj = im
            for i in iter_obj:
                imr = cv2.resize(i, dims)
                ft  = cv2.cvtColor(imr, cv2.COLOR_RGB2GRAY)
                if fttype == FeatureType.PATCHNORM:
                    ft = patchNormaliseImage(ft, 8)
                elif fttype == FeatureType.ROLLNORM:
                    ft = rollNormaliseImage(ft, 8)
                ft_ready_list.append(ft.flatten())
            if len(ft_ready_list) == 1:
                ft_ready = ft_ready_list[0]
            else:
                ft_ready = np.stack(ft_ready_list)
        elif fttype == FeatureType.HYBRIDNET:
            ft_ready = nn_hybrid.getFeat(im, use_tqdm=use_tqdm)
        elif fttype == FeatureType.NETVLAD:
            ft_ready = nn_netvlad.getFeat(im, use_tqdm=use_tqdm)
        else:
            raise Exception("[getFeat] fttype not recognised.")
        ft_list.append(ft_ready)
    if len(ft_list) == 1: 
        return ft_list[0]
    return ft_list

def patchNormaliseImage(img, patchLength):
# TODO: vectorize
# take input image, divide into regions, normalise
# returns: patch normalised image

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

            img2[iStart:iEnd, jStart:jEnd] = img1[iStart:iEnd, jStart:jEnd] - mean1 # offset remove mean
            if std1 == 0:
                std1 = 0.1
            img2[iStart:iEnd, jStart:jEnd] /= std1 # crush by std

    return img2   

def rollNormaliseImage(img, kernel_size):
# take input image and use a rolling kernel to noramlise
# returns: rolling-kernel-normalised image

    img1            = img.astype(float)
    img2            = img1.copy()
    
    if kernel_size == 1: # single pixel; already p-n'd
        return img2
    
    k_options       = list(range(-kernel_size,kernel_size+1,1))
    rolled_stack    = np.dstack([np.roll(np.roll(img2,i,0),j,1) for j in k_options for i in k_options])
    rollnormed      = np.mean(rolled_stack, 2) / np.std(rolled_stack, 2)

    return rollnormed  