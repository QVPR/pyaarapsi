# robotrun.py
'''
RobotRun Class
'''
from __future__ import annotations
import os
import copy
from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from pyaarapsi.vpred.visual_preproc import processImageDatasetFiltered
from pyaarapsi.vpr_simple.vpr_dataset_tool import VPRDatasetProcessor

def get_image_array(feature_array, dataset_params: dict, img_topic: Optional[str] = None
                    ) -> NDArray:
    '''
    Guarantee safe access of image topic array, using output of get_feature_array()
    '''
    if not len(dataset_params['img_topics']) == 1:
        if img_topic is None:
            raise ValueError("dataset has multiple image topics, but an img_topic specification"
                                " has not been provided.")
        else:
            assert img_topic in dataset_params['img_topics']
            img_index = dataset_params['img_topics'].index(img_topic)
            return feature_array[:, img_index, :]
    else:
        return feature_array

def get_feature_array(dataset_in: dict, dataset_params: dict, feature_type: Optional[str] = None
                    ) -> NDArray:
    '''
    Guarantee safe access of feature array
    '''
    if not len(dataset_params['ft_types']) == 1:
        if feature_type is None:
            raise ValueError("dataset has multiple feature types, but a ft_type specification"
                                " has not been provided.")
        else:
            assert feature_type in dataset_params['ft_types']
            return dataset_in[feature_type]
    else:
        return dataset_in[dataset_params['ft_types'][0]]

def get_odometry_arrays(dataset_in: dict, dataset_params: dict, odom_topic: Optional[str] = None
                        ) -> Tuple[NDArray, NDArray, NDArray]:
    '''
    Guarantee safe access of odometry arrays
    '''
    assert dataset_in['px'].ndim == dataset_in['py'].ndim == dataset_in['pw'].ndim
    if not dataset_in['pw'].ndim == 1:
        if odom_topic is None:
            raise ValueError("dataset has multiple odometries, but an odom_topic specification"
                                " has not been provided.")
        else:
            assert odom_topic in dataset_params['odom_topic']
            odom_index = dataset_params['odom_topic'].index(odom_topic)
            return dataset_in['px'][:, odom_index], dataset_in['py'][:, odom_index], \
                    dataset_in['pw'][:, odom_index]
    else:
        return dataset_in['px'], dataset_in['py'], dataset_in['pw']

def extract_imagenames(folder_name):
    '''
    TODO
    '''
    img_list = np.sort(os.listdir(folder_name))
    img_list = [os.path.join(folder_name,f) for f in img_list]
    return img_list

def make_odo_filenames(folder_name,imglist):
    '''
    TODO
    '''
    odo_filenames=[]
    for _,image_names in enumerate(imglist):
        odo_filenames.append(image_names[len(folder_name):\
                                         len(folder_name)+len('frame_id_000001')] + '.csv')
    return odo_filenames

def find_gaps(xy):
    '''
    TODO
    '''
    num_pts=len(xy)
    gaps = np.zeros(num_pts-1)
    for j, _ in enumerate(gaps):
        gaps[j] = np.linalg.norm(xy[j+1]-xy[j])
    return gaps

def find_distance_travelled(xy):
    '''
    TODO
    '''
    num_pts=len(xy)
    gaps=find_gaps(xy)
    distance_travelled=0
    path_distance=np.zeros(num_pts)
    for j, _ in enumerate(gaps):
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
    """
    Class for processing robot runs through an environment
    """
    def __init__(self,folder,npz_setup=False):
        """Constructs the attributes of a RobotRun object.
        
        Parameters
        ----------
        """
        self.npz_setup = npz_setup #temp fix
        self.description = ""
        self.folder = ""
        self.odo_folder = ""
        self.frame_size = -1
        self.x: NDArray = None
        self.y: NDArray = None
        self.xy: NDArray = None
        self.yaw: NDArray = None
        self.gaps: list = None
        if not npz_setup:
            self.folder = folder
            self.imlist = extract_imagenames(folder)
            self.imgnum = len(self.imlist)
            self.description = folder
        self.sample_rate = None
        self.features = np.array([])
        self.odo_list = []
        self.has_odom_data = False
        self.vpr_technique = ""
        self.timestamp = None
        self.along_path_distance = None
        self.npz_dictionary = None

    def from_dataset_dictionaries(self, dataset_in: dict, dataset_params: dict, bag_path: str,
                                  feature_type: Optional[str], odom_topic: Optional[str],
                                  img_topic: Optional[str]) -> RobotRun:
        '''
        TODO
        '''
        self.folder          = ""
        self.npz_setup       = True
        dataset_safe = copy.deepcopy(dataset_in)
        params_safe = copy.deepcopy(dataset_params)
        features = get_feature_array(dataset_in=dataset_safe, dataset_params=params_safe,
                                     feature_type=feature_type)
        single_feature_array = get_image_array(feature_array=features,
                                               dataset_params=params_safe, img_topic=img_topic)
        self.set_features(single_feature_array, 64)
        px, py, pw = get_odometry_arrays(dataset_in=dataset_safe, dataset_params=params_safe,
                                         odom_topic=odom_topic)
        self.set_xy(np.c_[px, py])
        self.set_yaw(np.degrees(pw))
        self.set_sample_rate(params_safe['sample_rate'])
        self.set_description(bag_path)
        self.set_npz_dictionary(params_safe)
        return self

    def from_dataset_processor(self, vprdp: VPRDatasetProcessor, feature_type: Optional[str],
                               odom_topic: Optional[str], img_topic: Optional[str]):
        '''
        TODO
        '''
        dataset_in = copy.deepcopy(vprdp.get_data())
        dataset_params = copy.deepcopy(vprdp.get_dataset_params())
        self.from_dataset_dictionaries(dataset_in=dataset_in,
                                       dataset_params=dataset_params,
                                       bag_path=vprdp.get_bag_path(), feature_type=feature_type,
                                       img_topic=img_topic, odom_topic=odom_topic)
        return self

    def set_npz_dictionary(self, _npz_dict: dict):
        '''
        TODO
        '''
        self.npz_dictionary = copy.deepcopy(_npz_dict)

    def get_npz_dictionary(self):
        '''
        TODO
        '''
        if self.npz_dictionary is None:
            raise AttributeError("Dictionary has not been assigned.")
        return copy.deepcopy(self.npz_dictionary)

    def set_features(self,features,size):
        '''
        Manually assign feature matrix
        '''
        self.features = features
        self.imgnum = features.shape[0]
        self.frame_size =size

    def set_description(self, description):
        '''
        Save a description for the run
        '''
        self.description = description

    def set_sample_rate(self, rate):
        '''
        Set image sampling rate in Hertz
        '''
        self.sample_rate = rate

    def image(self, number):
        '''
        Returns a single image from the run
        '''
        return mpimg.imread(self.imlist[number])

    def extract_sad_features(self, feature_type="downsampled_raw", size=64):
        '''
        Extracts SAD features
        '''
        self.frame_size = size
        self.features = processImageDatasetFiltered(self.folder, self.imlist, feature_type, size)[1]
        self.vpr_technique = 'SAD'

    def feature(self,number):
        '''
        Returns a single feature vector
        '''
        return self.features[number,:]

    def feature_image(self,number):
        '''
        Returns a feature vector reshaped into an image
        '''
        if self.features.size == 0:
            print('Error: feature_image: no features')
            return
        else:
            size=int((len(self.feature(number)))**(1/2)) # frame size of reshaped feature vector
            return self.feature(number).reshape((size,size))

    def set_xy(self, xy, num: int = None):
        '''
        Set odometry data for images as [x,y] coordinates
        Either for the complete array, or a single frame
        TODO: Add error check here
        '''
        if num is None:
            assert len(xy) == self.imgnum
            self.xy=xy
            self.x=xy[:,0]
            self.y=xy[:,1]
        else:
            assert isinstance(num, int) and (num <= self.imgnum)
            self.xy[num]=xy
            self.x[num]=xy[0]
            self.y[num]=xy[1]
        self.has_odom_data = True
        self.find_along_path_distances()

    def set_yaw(self, yaw, num: int = None):
        '''
        Set a yaw value (quasi heading) in the odometry data for each image
        Either for the complete array, or a single frame
        '''
        if num is None:
            assert len(yaw) == self.imgnum
            self.yaw = yaw
        else:
            assert isinstance(num, int) and (num <= self.imgnum)
            self.yaw[num]=yaw

    def wrap_yaw(self):
        '''
        TODO
        '''
        tmp_heading = wrap_heading(self.yaw)
        self.yaw = tmp_heading
        return self.yaw

# TODO: Add timestamps - need to check timestamp length matches other data, and deal with truncation
#    def set_timestamp(timestamps):
#        if len(timestamps) == self.imgnum:
#            self.timestamp = timestamps

    def extract_xy(self, odo_folder):
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
        self.has_odom_data = True
        self.find_along_path_distances()

    def truncate(self, start, end, verbose=True):
        '''
        Remove the section that is prior to the "start" index, and after the "end" index given
        '''
        if self.has_odom_data: # update odometry data if it is defined
            _len = len(self.xy)
            self.xy=self.xy[start:end+1] # (note cannot use slices here, need to use indexing)
            self.x=self.xy[:,0]
            self.y=self.xy[:,1]
            self.yaw=self.yaw[start:end+1]
        if len(self.odo_list) > 0: # update odometry filenames if they exist
            self.odo_list=self.odo_list[start:end+1]
        if self.features.size > 0: # update feature matrix (also use indices not slicing)
            self.features=self.features[start:end+1]
            self.imgnum=self.features.shape[0]
        if not self.npz_setup:
            self.imlist=self.imlist[start:end+1] # update filenames
            self.imgnum = len(self.imlist)
        self.find_along_path_distances()
        if not verbose:
            return
        if self.has_odom_data:
            print(f'[RobotRun: truncate] Reduced from {_len} to {self.imgnum} images.')
        else:
            print(f'[RobotRun: truncate] Reduced to {self.imgnum} images.')

    def subsample(self):
        '''
        TODO
        '''
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
        if not self.has_odom_data:
            print('Error: RobotRun.find_path_distances: run has no odometry data.')
            return
        gaps=np.zeros(self.imgnum-1)
        distance_travelled=0
        along_path_distance=np.zeros(self.imgnum)
        for j, _ in enumerate(gaps):
            gaps[j] = np.linalg.norm(self.xy[j+1]-self.xy[j])
            distance_travelled = distance_travelled + gaps[j]
            along_path_distance[j+1] = distance_travelled
        self.gaps = gaps
        self.along_path_distance = along_path_distance
        return

    def __repr__(self):  # Return a string containing a printable representation of an object.
        return f"{self.__class__.__name__}(description={self.description})"
