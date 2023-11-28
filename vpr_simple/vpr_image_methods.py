import cv2
import numpy as np
import copy

def grey2dToColourMap(matrix, colourmap=cv2.COLORMAP_JET, dims=None):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    matnorm = (((matrix - min_val) / (max_val - min_val)) * 255).astype(np.uint8)
    if not (dims is None):
        matnorm = cv2.resize(matnorm, dims, interpolation=cv2.INTER_AREA)
    mat_rgb = cv2.applyColorMap(matnorm, colourmap)
    return mat_rgb

def label_image(img_in, text: str, position, colour, border=(0,0,0), make_copy: bool = True, 
                scale: float = 1, thickness: int = 2, border_thickness: int = 7):
# Write text at position position, with colour and black border on img_in
    if make_copy:
        img = img_in.copy()
    else:
        img = img_in
    if not (border is None):
        img = cv2.putText(img, text, org=position, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale, \
                                    color=border, thickness=border_thickness, lineType=cv2.LINE_AA)
    # Colour inside:
    img = cv2.putText(img, text, org=position, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale, \
                                color=colour, thickness=thickness, lineType=cv2.LINE_AA)
    return img

def convert_img_to_uint8(img_in, resize=None, dstack=True, make_copy=True):
    if make_copy:
        img = img_in.copy()
    else:
        img = img_in

    if not type(img.flatten()[0]) == np.uint8:
        _min      = np.min(img)
        _max      = np.max(img)
        _img_norm = (img - _min) / (_max - _min)
        img       = np.array(_img_norm * 255, dtype=np.uint8)
    if not resize is None:
        if len(img.shape) == 1: # vector
            img = np.reshape(img, (int(np.sqrt(img.shape[0])),)*2)
        img = cv2.resize(img, resize, interpolation = cv2.INTER_AREA)
    if dstack: return np.dstack((img,)*3)
    return img

def apply_icon(img_in, position, icon, make_copy=True):
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
    
    icon_mask_inv       = cv2.inRange(src=icon, lowerb=np.array([50,50,50]), upperb=np.array([255,255,255]))  # get background, white region (https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html)
    icon_mask           = 255 - icon_mask_inv                           # invert background to get shape
    icon_mask_stack_inv = cv2.merge([icon_mask_inv, icon_mask_inv, icon_mask_inv]) / 255    # stack into rgb layers, convert to range 0.0 -> 1.0
    icon_mask_stack     = cv2.merge([icon_mask, icon_mask, icon_mask]) / 255                # stack into rgb layers, convert to range 0.0 -> 1.0
    
    opacity_icon        = 0.8 # 80%

    # Create new slice of image with icon on-top, three pieces, sum:
    # 1. image outside of icon; opacity = 1
    # 2. icon;                  opacity = opacity_icon
    # 3. image covered by icon; opacity = (1-opacity_icon)
    img_slice = (icon_mask_stack_inv * img_slice) + \
                (icon_mask_stack * icon) * (opacity_icon) + \
                (icon_mask_stack * img_slice) * (1-opacity_icon)
    
    # Insert slice back into original image:
    img[start_row:end_row, start_col:end_col, :] = img_slice

    return img

def makeImage(query_raw, match_raw, icon_dict):
# Produce image to be published via ROS that has a side-by-side style of match (left) and query (right)
# Query image comes in via cv2 variable query_raw
# Match image comes in from ref_dict

    match_img   = convert_img_to_uint8(match_raw, resize=(500,500), dstack=(not len(match_raw.shape) == 3))
    query_img   = convert_img_to_uint8(query_raw, resize=(500,500), dstack=(not len(query_raw.shape) == 3))
    
    match_img_lab = label_image(match_img, "Reference", (20,40), (100,255,100))
    query_img_lab = label_image(query_img, "Query",     (20,40), (100,255,100))

    try:
        icon_to_use = icon_dict['icon'] # accelerate access
        icon_size   = icon_dict['size'] # ^
        icon_dist   = icon_dict['dist'] # ^
        if icon_size > 0:
            # Add Icon:
            img_slice = query_img_lab[-1 - icon_size - icon_dist:-1 - icon_dist, -1 - icon_size - icon_dist:-1 - icon_dist, :]
            # https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
            icon_mask_inv = cv2.inRange(src=icon_to_use, lowerb=np.array([50,50,50]), upperb=np.array([255,255,255])) # get border (white)
            icon_mask = 255 - icon_mask_inv # get shape
            icon_mask_stack_inv = cv2.merge([icon_mask_inv, icon_mask_inv, icon_mask_inv]) / 255 # stack into rgb layers, binary image
            icon_mask_stack = cv2.merge([icon_mask, icon_mask, icon_mask]) / 255 # stack into rgb layers, binary image
            opacity_icon = 0.8 # 80%
            # create new slice with appropriate layering
            img_slice = (icon_mask_stack_inv * img_slice) + \
                        (icon_mask_stack * icon_to_use) * (opacity_icon) + \
                        (icon_mask_stack * img_slice) * (1-opacity_icon)
            query_img_lab[-1 - icon_size - icon_dist:-1 - icon_dist, -1 - icon_size - icon_dist:-1 - icon_dist, :] = img_slice
    except:
        pass # dictionary bad, probably usage of groundtruth disable

    return np.concatenate((match_img_lab, query_img_lab), axis=1)