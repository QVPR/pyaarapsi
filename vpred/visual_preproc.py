# This file contains functions to pre-process images before using in VPR algorithms
#
# Note this code is from QVCR repository

import os
import numpy as np
import cv2
from tqdm import tqdm
import math
from scipy.spatial.distance import cdist

def loadImg(imPath,reso=None):
    im = cv2.imread(imPath)[:,:,::-1] #is this to transfer from RGB to BGR ??? 
    if reso is not None:
        im = cv2.resize(im,reso)
    return im

# def processImageDataset(path,ftType,size):
#     print("Computing features: {} for image path: {} ...".format(ftType, path))
#     # create 
#     imgList = np.sort(os.listdir(path))
#     imgList = [os.path.join(path,f) for f in imgList]
    
#     # Init list of features
#     feats = []
    
#     # Cycle through the list of images
#     for i, imPath in tqdm(enumerate(imgList)):  # tqdm provides a progress bar
#         frame = loadImg(imPath)                 # load image without changing resolution
#         feat = getFeat(frame,ftType,size)       # modified! process image by returning 1 x 4096 flattened greyscale of image ## Modified to use "size" passed
#         feats.append(feat)                      # add flattened downscaled, greyscale image to the list
#     feats = np.array(feats)
#     print(feats.shape)
#     return imgList, feats  #imgList is list of image names with full pathname, feats is an array of flattened downscaled, greyscale images

def processImageDatasetFiltered(path,imgList,ftType,size):
    print("Computing features: {} ...".format(ftType))
    
    # Init list of features
    feats = []
    
    # Cycle through the list of images
    for i, imPath in tqdm(enumerate(imgList)):  # tqdm provides a progress bar
        frame = loadImg(imPath)                 # load image without changing resolution
        feat = getSADFeats(frame,ftType,size)       # modified! process image by returning 1 x 4096 flattened greyscale of image ## Modified to use "size" passed
        feats.append(feat)                      # add flattened downscaled, greyscale image to the list
    feats = np.array(feats)
    print(feats.shape)
    return imgList, feats  #imgList is list of image names with full pathname, feats is an array of flattened downscaled, greyscale images


# Note this code is from QVCR repository
def patchNormImage(img1,patchLength):
    numZeroStd = []
    img1 = img1.astype(float)
    img2 = img1.copy()
    imgMask = np.ones(img1.shape,dtype=bool)
    
    if patchLength == 1:
        return img2

    for i in range(img1.shape[0]//patchLength):
        iStart = i*patchLength
        iEnd = (i+1)*patchLength
        for j in range(img1.shape[1]//patchLength):
            jStart = j*patchLength
            jEnd = (j+1)*patchLength
            tempData = img1[iStart:iEnd, jStart:jEnd].copy()
            mean1 = np.mean(tempData)
            std1 = np.std(tempData)
            tempData = (tempData - mean1)
            if std1 == 0:
                std1 = 0.1 
                numZeroStd.append(1)
                imgMask[iStart:iEnd,jStart:jEnd] = np.zeros([patchLength,patchLength],dtype=bool)
            tempData /= std1
            img2[iStart:iEnd, jStart:jEnd] = tempData.copy()
    return img2, imgMask

# Note this code is from QVCR repository
def getMatchInds(ft_ref,ft_qry,topK=20,metric='euclidean'):
    dMat = cdist(ft_ref,ft_qry,metric)
    mInds = np.argsort(dMat,axis=0)[:topK] # shape: K x ft_qry.shape[0]
    return mInds

# Note this code is from QVCR repository - modified to allow downsampling size to be modified
#
#  getFeat returns a 1D array, which is the flattened [1 x 4096] greyscale image that is reduced to 64x64
#
def getSADFeats(im,ftType,size=64): # Modified to pass size of downsized image
    if ftType == "downsampled_raw":
        im = cv2.resize(im,(size,size))                # downsample to change resolution
        ft = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)   # change from RGB colour to greyscale
    elif ftType == "downsampled_patchNorm":
        im = cv2.resize(im,(size,size))
        im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        ft,_ = patchNormImage(im,size)
    return ft.flatten()

def processImageDatasetORB(path,size1=320,size2=240):
    """Process images and return imgList, feats, keypts
    """
    print("Computing features: {} for image path: {} ...".format('ORB', path))
    # create 
    imgList = np.sort(os.listdir(path))
    imgList = [os.path.join(path,f) for f in imgList]
    
    orb = cv2.ORB_create()
    
    # Init list of features
    feats = []
    keypts = []
    
    # Cycle through the list of images
    for i, imPath in tqdm(enumerate(imgList)):  # tqdm provides a progress bar
        frame = loadImg(imPath)                 # load image without changing resolution
        frame = cv2.resize(frame,(size1,size2))       # downsample
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)   # change to grayscale
        keypoints, descriptor = orb.detectAndCompute(frame, None)
        feats.append(descriptor)  # add descriptors to the feature list
        keypts.append(keypoints)  # add keypoints to the keypoint list
    feats = np.array(feats,dtype='object')
    keypts = np.array(keypts,dtype='object')
    #print(feats.shape)
    return imgList, feats, keypts  #imgList is list of image names with full pathname, feats is an array of flattened downscaled, greyscale images