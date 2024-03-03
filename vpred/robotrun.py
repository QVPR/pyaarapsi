# robotrun.py

import numpy as np
import os
import copy
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from .visual_preproc import processImageDatasetFiltered
from ..vpr_simple.vpr_dataset_tool import VPRDatasetProcessor

def extract_imagenames(folder_name):
    imgList = np.sort(os.listdir(folder_name))
    imgList = [os.path.join(folder_name,f) for f in imgList]
    return imgList

def make_odo_filenames(folder_name,imglist):
    odo_filenames=[]
    for i,image_names in enumerate(imglist):
        odo_filenames.append(image_names[len(folder_name):len(folder_name)+len('frame_id_000001')] + '.csv')
    return odo_filenames

def find_gaps(xy):
    num_pts=len(xy)
    gaps=np.zeros(num_pts-1)
    for j in range(len(gaps)):
        gaps[j]=np.linalg.norm(xy[j+1]-xy[j])
    return gaps

def find_distance_travelled(xy):
    num_pts=len(xy)
    gaps=find_gaps(xy)
    distance_travelled=0
    path_distance=np.zeros(num_pts)
    for j in range(len(gaps)):
        distance_travelled=distance_travelled + gaps[j]
        path_distance[j+1]=distance_travelled
    return path_distance

def wrap_heading(heading):
    '''
    Wrap heading in degrees to avoid bouncing around -180 to +180
    '''
    new_heading = np.zeros_like(heading)
    for i,h in enumerate(heading):
        if h < -175:
            new_heading[i]=h+360
        else:
            new_heading[i]=h
    return new_heading

class RobotRun:
    """Class for processing robot runs through an environment"""
    
    def __init__(self,folder,npz_setup=False):
        """Constructs the attributes of a RobotRun object.
        
        Parameters
        ----------
        """
        self.npz_setup = npz_setup #temp fix
        self.description = ""
        if npz_setup == False:
            self.folder = folder
            self.imlist = extract_imagenames(folder)
            self.imgnum = len(self.imlist)
            self.description = folder
        self.sample_rate = None
        self.features = np.array([])
        self.odo_list = []
        self.GEO_TAGGED = False
        self.vpr_technique = ""
        self.timestamp = None
        self.along_path_distance = None
        self.npz_dictionary = None

    def from_dataset_dictionaries(self, dataset_in: dict, ip_dict: dict, bag_path: str):
        assert len(ip_dict['ft_types']) == 1
        self.folder     = ""
        self.npz_setup  = True
        self.set_features(dataset_in[ip_dict['ft_types'][0]], size=64)
        self.set_xy(np.c_[dataset_in['px'], dataset_in['py']])
        self.set_yaw(np.degrees(dataset_in['pw']))
        self.set_sample_rate=ip_dict['sample_rate']
        self.set_description(bag_path)
        self.set_npz_dictionary(ip_dict)
        return self

    def from_dataset_processor(self, vprdp: VPRDatasetProcessor):
        dataset_in = vprdp.dataset['dataset']
        ip_dict = vprdp.get_dataset_params()
        self.from_dataset_dictionaries(dataset_in=dataset_in, ip_dict=ip_dict, bag_path=vprdp.get_bag_path())
        return self

    def set_npz_dictionary(self, _npz_dict: dict):
        self.npz_dictionary = copy.deepcopy(_npz_dict)

    def get_npz_dictionary(self):
        if self.npz_dictionary is None: raise Exception("Dictionary has not been assigned.")
        return copy.deepcopy(self.npz_dictionary)
    
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
        self.find_along_path_distances()
        
    def set_yaw(self, yaw, num=None):
        '''
        Set a yaw value (quasi heading) in the geo-tag for each image
        Either for the complete array, or a single frame
        '''
        if num == None:
            # if len(yaw) != self.imgnum:
            #     print('Error: set_yaw requires length of yaw vector to equal number of images in run');
            # else:
            self.yaw=yaw
        else: # TODO: check num is an integer <= self.imgnum
            self.yaw[num]=yaw
            
    def wrap_yaw(self):
        tmp_heading = wrap_heading(self.yaw)
        self.yaw = tmp_heading
        return self.yaw

# TODO: Add timestamps - need to check timestamp length matches other data, and deal with truncation
#    def set_timestamp(timestamps):
#        if len(timestamps) == self.imgnum:
#            self.timestamp = timestamps

            
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
        self.find_along_path_distances()
        
    def truncate(self, start, end, verbose=True):
        '''
        Remove the section that is prior to the "start" index, and after the "end" index given
        '''
        if self.GEO_TAGGED:                      # update geotags if they are defined
            _len = len(self.xy)
            self.xy=self.xy[start:end+1]  # (note cannot use slices here, need to use indexing)
            self.x=self.xy[:,0]
            self.y=self.xy[:,1]
            self.yaw=self.yaw[start:end+1]
        if len(self.odo_list) > 0:               # update odometry filenames if they exist
            self.odo_list=self.odo_list[start:end+1]
        if self.features.size > 0:               # update feature matrix (also use indices not slicing)
            self.features=self.features[start:end+1]
            self.imgnum=self.features.shape[0]
        if self.npz_setup==False:
            self.imlist=self.imlist[start:end+1]     # update filenames
            self.imgnum = len(self.imlist)
        self.find_along_path_distances()
        if not verbose: return
        if self.GEO_TAGGED:
            print('[RobotRun: truncate] Reduced from {0} to {1} images.'.format(_len, self.imgnum))
        else:
            print('[RobotRun: truncate] Reduced to {0} images.'.format(self.imgnum))
        
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

    def find_along_path_distances(self):
        '''
        Return a vector containing along-path distance travelled along the run
        '''
        if self.GEO_TAGGED == False:
            print('Error: RobotRun.find_path_distances: run is not geotagged')
            return
        gaps=np.zeros(self.imgnum-1)
        distance_travelled=0
        along_path_distance=np.zeros(self.imgnum)
        for j in range(len(gaps)):
            gaps[j]=np.linalg.norm(self.xy[j+1]-self.xy[j])
            distance_travelled=distance_travelled + gaps[j]
            along_path_distance[j+1]=distance_travelled
        self.gaps = gaps
        self.along_path_distance = along_path_distance
        return

    def __repr__(self):  # Return a string containing a printable representation of an object.
        return f"{self.__class__.__name__}(description={self.description})"
