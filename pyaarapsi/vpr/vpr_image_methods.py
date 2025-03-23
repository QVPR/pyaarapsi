#!/usr/bin/env python3
'''
Collection of tools for images in VPR works
'''
#pylint: disable=E0611
from cv2 import COLORMAP_JET as cv_COLORMAP_JET, applyColorMap as cv_applyColorMap, \
    putText as cv_putText, FONT_HERSHEY_SIMPLEX as cv_FONT_HERSHEY_SIMPLEX, LINE_AA as cv_LINE_AA, \
    resize as cv_resize, INTER_AREA as cv_INTER_AREA, inRange as cv_inRange, merge as cv_merge, \
    imread as cv_imread
#pylint: enable=E0611
import numpy as np

def load_img(img_path, resolution=None):
    '''
    Load an image from a path; resize if desired
    '''
    im = cv_imread(img_path)[:,:,::-1]
    if resolution is not None:
        im = cv_resize(im, resolution)
    return im

def greyscale_to_colourmap(matrix, colourmap=cv_COLORMAP_JET, dims=None):
    '''
    Apply a colour map to a greyscale image
    '''
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    matnorm = (((matrix - min_val) / (max_val - min_val)) * 255).astype(np.uint8)
    if dims is not None:
        matnorm = cv_resize(matnorm, dims, interpolation=cv_INTER_AREA)
    mat_rgb = cv_applyColorMap(matnorm, colourmap)
    return mat_rgb

def label_image(img_in, text: str, position, colour, border=(0,0,0), make_copy: bool = True,
                scale: float = 1, thickness: int = 2, border_thickness: int = 7):
    '''
    Write text at position position, with colour and black border on img_in
    '''
    if make_copy:
        img = img_in.copy()
    else:
        img = img_in
    if border is not None:
        img = cv_putText(img, text, org=position, fontFace=cv_FONT_HERSHEY_SIMPLEX, \
                            fontScale=scale, color=border, thickness=border_thickness, \
                            lineType=cv_LINE_AA)
    # Colour inside:
    img = cv_putText(img, text, org=position, fontFace=cv_FONT_HERSHEY_SIMPLEX, fontScale=scale, \
                                color=colour, thickness=thickness, lineType=cv_LINE_AA)
    return img

def convert_img_to_uint8(img_in, resize=None, dstack=True, make_copy=True):
    '''
    Bound and convert an image as uint8 values
    '''
    if make_copy:
        img = img_in.copy()
    else:
        img = img_in
    #
    if not isinstance(img.flatten()[0], np.uint8):
        _min      = np.min(img)
        _max      = np.max(img)
        _img_norm = (img - _min) / (_max - _min)
        img       = np.array(_img_norm * 255, dtype=np.uint8)
    #
    if not resize is None:
        if len(img.shape) == 1: # vector
            img = np.reshape(cv_resize(img.astype('float32'), (1,4096),
                                       interpolation = cv_INTER_AREA), (64,64))
        img = cv_resize(img, resize, interpolation = cv_INTER_AREA)
        if not isinstance(img_in.flatten()[0], np.uint8):
            img = np.round(img)
            img[img > 255] = 255
            img[img < 0] = 0
            img.astype(np.uint8)
    if dstack:
        return np.dstack((img,)*3)
    return img

def apply_icon(img_in, position, icon, make_copy=True):
    '''
    Insert an icon into an image
    '''
    if make_copy:
        img = img_in.copy()
    else:
        img = img_in

    size_y, size_x, _   = icon.shape
    pos_x, pos_y        = position

    start_col, end_col  = (pos_x, pos_x + size_x)
    start_row, end_row  = (pos_y, pos_y + size_y)

    # Extract slice of image to insert icon on-top:
    img_slice           = img[start_row:end_row, start_col:end_col, :]
    # Mask calculations
    # get background, white region:
    # (https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html)
    icon_mask_inv       = cv_inRange(src=icon, lowerb=np.array([50,50,50]),
                                     upperb=np.array([255,255,255]))
    # invert background to get shape:
    icon_mask           = 255 - icon_mask_inv
    # stack into rgb layers, convert to range 0.0 -> 1.0:
    icon_mask_stack_inv = cv_merge([icon_mask_inv, icon_mask_inv, icon_mask_inv]) / 255
    icon_mask_stack     = cv_merge([icon_mask, icon_mask, icon_mask]) / 255
    # Create new slice of image with icon on-top, three pieces, sum:
    # 1. image outside of icon; opacity = 1
    # 2. icon;                  opacity = opacity_icon
    # 3. image covered by icon; opacity = (1-opacity_icon)
    opacity_icon = 0.8 # 80%
    img_slice = (icon_mask_stack_inv * img_slice) + \
                (icon_mask_stack * icon) * (opacity_icon) + \
                (icon_mask_stack * img_slice) * (1-opacity_icon)
    # Insert slice back into original image:
    img[start_row:end_row, start_col:end_col, :] = img_slice
    return img

def make_image(query_raw, match_raw, icon_dict):
    '''
    Produce image to be published via ROS that has a side-by-side style of match (left) and 
    query (right)
    Query image comes in via cv2 variable query_raw
    Match image comes in from ref_dict
    '''
    match_img = convert_img_to_uint8(match_raw, resize=(500,500), dstack=len(match_raw.shape) != 3)
    query_img = convert_img_to_uint8(query_raw, resize=(500,500), dstack=len(query_raw.shape) != 3)
    # Make labels:
    match_img_lab = label_image(match_img, "Reference", (20,40), (100,255,100))
    query_img_lab = label_image(query_img, "Query",     (20,40), (100,255,100))
    # Prepare an icon:
    icon_to_use = icon_dict['icon'] # accelerate access
    icon_size   = icon_dict['size'] # ^
    icon_dist   = icon_dict['dist'] # ^
    if icon_size > 0:
        # Add Icon:
        img_slice = query_img_lab[-1 - icon_size - icon_dist : -1 - icon_dist,
                                  -1 - icon_size - icon_dist : -1 - icon_dist, :]
        # https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
        icon_mask_inv = cv_inRange(src=icon_to_use, lowerb=np.array([50,50,50]),
                                    upperb=np.array([255,255,255])) # get border (white)
        icon_mask = 255 - icon_mask_inv # get shape
        # stack into rgb layers, binary image:
        icon_mask_stack_inv = cv_merge([icon_mask_inv, icon_mask_inv, icon_mask_inv]) / 255
        icon_mask_stack = cv_merge([icon_mask, icon_mask, icon_mask]) / 255
        # create new slice with appropriate layering and opacity:
        opacity_icon = 0.8 # 80%
        img_slice = (icon_mask_stack_inv * img_slice) + \
                    (icon_mask_stack * icon_to_use) * (opacity_icon) + \
                    (icon_mask_stack * img_slice) * (1-opacity_icon)
        query_img_lab[-1 - icon_size - icon_dist : -1 - icon_dist,
                      -1 - icon_size - icon_dist : -1 - icon_dist, :] = img_slice
    return np.concatenate((match_img_lab, query_img_lab), axis=1)
