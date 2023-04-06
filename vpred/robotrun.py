# robotrun.py

import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from visual_preproc import *

def extract_imagenames(folder_name):
    imgList = np.sort(os.listdir(folder_name))
    imgList = [os.path.join(folder_name,f) for f in imgList]
    return imgList

def make_odo_filenames(folder_name,imglist):
    odo_filenames=[]
    for i,image_names in enumerate(imglist):
        odo_filenames.append(image_names[len(folder_name):len(folder_name)+len('frame_id_000001')] + '.csv')
    return odo_filenames

class RobotRun:
    """Class for processing robot runs through an environment"""
    
    def __init__(self,folder,npz_setup=False):
        """Constructs the attributes of a RobotRun object.
        
        Parameters
        ----------
        """
        self.npz_setup = npz_setup #temp fix
        if npz_setup == False:
            self.folder = folder
            self.imlist = extract_imagenames(folder)
            self.imgnum = len(self.imlist)
        self.description = ""
        self.sample_rate = None
        self.features = np.array([])
        self.odo_list = []
        self.GEO_TAGGED = False
        self.vpr_technique = ""
    
    def set_features(self,features,size):
        '''Manually assign feature matrix'''
        self.features = features
        self.imgnum = features.shape[0]
        self.frame_size =size
    
    def set_description(self, description):
        '''Save a description for the run'''
        self.description = description
    
    def set_sample_rate(self, rateHz):
        '''Set image sampling rate in Hertz'''
        self.sample_rate=rateHz
        
    def image(self, number):
        '''Returns a single image from the run'''
        return mpimg.imread(self.imlist[number])
    
    def extract_SAD_features(self, ftType="downsampled_raw", size=64):
        '''Extracts SAD features'''
        self.frame_size=size
        _,self.features = processImageDatasetFiltered(self.folder,self.imlist,ftType,size)
        self.vpr_technique = 'SAD'
        
    def feature(self,number):
        '''Returns a single feature vector'''
        return self.features[number,:]
    
    def feature_image(self,number):
        '''Returns a feature vector reshaped into an image'''
        if self.features.size == 0:
            print('Error: feature_image: no features')
            return
        else:
            size=int((len(self.feature(number)))**(1/2)) # frame size of reshaped feature vector
            return self.feature(number).reshape((size,size))
        
    def set_xy(self, xy, num=None):
        '''
        Set geotags for images as [x,y] coordinates
        Either for the complete array, or a single frame
        '''
        # TODO: Add error check here
        if num == None:
            self.xy=xy
            self.x=xy[:,0]
            self.y=xy[:,1]
        else: # TODO: check num is an integer <= self.imgnum
            self.xy[num]=xy
            self.x[num]=xy[0]
            self.y[num]=xy[1]
        self.GEO_TAGGED = True
            
    def extract_geotags(self, odo_folder):
        '''
        Extract x,y coordinates from a folder
        Each image will have a corresponding .csv file containing coordinates
        '''
        self.odo_folder=odo_folder
        self.odo_list=make_odo_filenames(self.folder,self.imlist)
        self.xy=np.zeros([self.imgnum,2],dtype='float')
        for i,fname in enumerate(self.odo_list):
            self.xy[i]=np.loadtxt(self.odo_folder+fname)[:2] # Assume .cvs format is x,y or x,y,z
        self.x=self.xy[:,0]
        self.y=self.xy[:,1]
        self.GEO_TAGGED = True
        
    def truncate(self, start, end):
        '''
        Remove the section that is prior to the "start" index, and after the "end" index given
        '''
        if self.GEO_TAGGED:                      # update geotags if they are defined
            self.xy=self.xy[np.r_[start:end+1]]  # (note cannot use slices here, need to use indexing)
            self.x=self.xy[:,0]
            self.y=self.xy[:,1]
        if len(self.odo_list) > 0:               # update odometry filenames if they exist
            self.odo_list=self.odo_list[start:end+1]
        if self.features.size > 0:               # update feature matrix (also use indices not slicing)
            self.features=self.features[np.r_[start:end+1]]
            self.imgnum=self.features.shape[0]
        if self.npz_setup==False:
            self.imlist=self.imlist[start:end+1]     # update filenames
            self.imgnum = len(self.imlist)
        print('truncate: run is now {0} images long'.format(self.imgnum))
        
    def subsample(self):
        #TODO
        # if self.sample_rate != None:
        #     Add subsampling return here
        # else:
        #     print('Error: subsample: sampling rate of original run is undefined')
        return
        
    def show_feature_image(self,number):
        '''
        Plot feature image
        '''
        if self.features.size==0:
            print('Error: feature_image: no features')
        else:
            fig,ax=plt.subplots()
            ax.imshow(self.feature_image(number),cmap='gray')
            ax.set_axis_off()
        return fig,ax
        
    def __repr__(self):  # Return a string containing a printable representation of an object.
        return f"{self.__class__.__name__}(description={self.description})"