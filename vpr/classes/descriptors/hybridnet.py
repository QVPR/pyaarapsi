#!/usr/bin/env python3
'''
HybridNet Container for feature extraction
'''
import os
import numpy as np
from tqdm.auto import tqdm
from cv2 import imread as cv_imread #pylint: disable=E0611
from PIL import Image
from pyaarapsi.vpr.classes.download_models import download_hybridnet_models, MODELS_DIR
from pyaarapsi.vpr.classes.descriptors.generic import GenericPlaceDataset, \
    PlaceDatasetError, DescriptorContainerError, DescriptorContainer

try:
    import caffe
    # must be done prior to importing caffe to suppress excessive logging:
    os.environ['GLOG_minloglevel'] = '2'
    CAFFE_OK = True
except ImportError:
    CAFFE_OK = False

class PlaceDataset(GenericPlaceDataset):
    '''
    HybridNet dataset handler
    '''
    def __init__(self, image_data, dims=None):
        super().__init__()
        self.images = image_data
        self.dims = dims
        self.getitem = None
        self.len = None
        if isinstance(self.images, list):
            # either contains PIL Images, str paths to images, or np.ndarray images
            self.len = lambda: len(self.images)
            if isinstance(self.images[0], Image.Image): # type list of PIL Images
                self.getitem = lambda index: np.array(self.images[index])
            elif isinstance(self.images[0], str): # type list of strs
                self.getitem = lambda index: cv_imread(self.images[index])
            elif isinstance(self.images[0], np.ndarray): # type list of np.ndarray images
                if len(self.images[0].shape) == 2: # greyscale image
                    self.getitem = lambda index: \
                        np.dstack((self.images[index],)*3).astype(np.uint8)
                else: # rgb image:
                    self.getitem = lambda index: \
                        self.images[index].astype(np.uint8)
            else:
                raise PlaceDatasetError("Input of type list but contains elements of unknown type. "
                                f"Type: {str(type(self.images[0]))}")
        elif isinstance(self.images, np.ndarray):
            self.len = lambda: \
                self.images.shape[0]
            if self.dims is None: # type np.ndarray of np.ndarray images
                if len(self.images[0].shape) == 2: # grayscale
                    self.getitem = lambda index: np.dstack((self.images[index],)*3).astype(np.uint8)
                else:
                    self.getitem = lambda index: \
                        self.images[index].astype(np.uint8)
            else: # type np.ndarray of np.ndarray flattened images
                self.getitem = lambda index: \
                    np.dstack((np.reshape(self.images[index], self.dims),)*3).astype(np.uint8)
        else:
            raise PlaceDatasetError("Input not of type list or np.ndarray. "
                                    f"Type: {str(type(self.images))}")
    #
    def __getitem__(self, index):
        '''
        Get items
        '''
        return self.getitem(index), index
    #
    def __len__(self):
        '''
        Length of dataset
        '''
        return self.len()
    #
    def __del__(self):
        '''
        Clean-up
        '''
        if self.len is not None:
            del self.len
        if self.getitem is not None:
            del self.getitem
        del self.images
        del self.dims

class HybridNetContainer(DescriptorContainer):
    '''
    Feature extraction for HybridNet
    '''
    def __init__(self, logger=print, cuda=False, target_layer='fc7_new', load=True, prep=True):
        self.cuda           = cuda
        self.logger         = logger
        self.target_layer   = target_layer
        # keep these features dim fixed as they need to match the network architecture
        # inside "HybridNet"
        self.layer_labs      = ['conv3', 'conv4', 'conv5', 'conv6' ,'pool1', 'pool2',
                                'fc7_new', 'fc8_new']
        self.layer_dims      = [64896, 64896, 43264, 43264, 69984, 43264, 4096, 2543]
        self.layer_dict      = dict(zip(self.layer_labs, self.layer_dims))
        self.loaded = False
        self.prepped = False
        self.net = None
        self.transformer = None
        if not CAFFE_OK:
            self.logger('Could not import caffe for HybridNet Container. Using the container '
                        'will trigger crashes. Please correct your caffe installation.')
            return
        if self.cuda:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        if load:
            self.load()
        if prep:
            self.prep()
    #
    def is_ready(self):
        '''
        Whether model can be used right now
        '''
        return self.prepped
    #
    def ready_up(self):
        '''
        Prepare for feature extraction
        '''
        if self.is_ready():
            return

        if not self.loaded:
            self.load()
        if not self.prepped:
            self.prep()
    #
    def load(self):
        '''
        Load model data
        '''
        model_file_path = os.path.join(MODELS_DIR, 'HybridNet.caffemodel')
        if not os.path.isfile(model_file_path):
            download_hybridnet_models()
        self.logger('Loading HybridNet model')
        self.net = caffe.Net(os.path.join(MODELS_DIR,'deploy.prototxt'),
                             model_file_path, caffe.TEST)
        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2,0,1))
        self.transformer.set_mean('data', np.load(os.path.join(MODELS_DIR, 'amosnet_mean.npy')
                                                  ).mean(1).mean(1)) # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]:
        self.transformer.set_raw_scale('data', 255)
        # the reference model has channels in BGR order instead of RGB:
        self.transformer.set_channel_swap('data', (2,1,0))
        self.net.blobs['data'].reshape(1,3,227,227)
        self.logger('HybridNet loaded')
        self.loaded = True
    #
    def __del__(self):
        '''
        Destroy all data, objects
        '''
        del self.logger
        del self.cuda
        del self.target_layer
        del self.layer_labs
        del self.layer_dims
        del self.layer_dict
        del self.loaded
        del self.prepped
        if self.net is not None:
            del self.net
        if self.transformer is not None:
            del self.transformer
    #
    def prep(self):
        '''
        Any prepwork, such as passing a default object through the container
        '''
        self.prepped = True
    #
    def clean_data_input(self, dataset_input, dims):
        '''
        Process input
        '''
        # either a single image path or a set (or a bad path has been provided):
        if isinstance(dataset_input, str):
            try:
                img = cv_imread(dataset_input)
                if img is None:
                    raise DescriptorContainerError('Image path is invalid')
                dataset_clean = PlaceDataset([dataset_input]) # handle single image
                # INPUTS: type list of strs
            except (DescriptorContainerError, PlaceDatasetError):
                try:
                    ref_filenames = [os.path.join(dataset_input, filename)
                                    for filename in sorted(os.listdir(dataset_input))
                                    if os.path.isfile(os.path.join(dataset_input, filename))]
                    dataset_clean = PlaceDataset(ref_filenames) # handle directory of images
                    # INPUTS: type list of strs
                except Exception as e:
                    raise DescriptorContainerError("Bad path specified; input is not an image or "
                                                   "directory path") from e
        elif isinstance(dataset_input, Image.Image):
            # handle PIL image:
            dataset_clean = PlaceDataset([dataset_input])
            # INPUTS: type list of PIL Images
        elif isinstance(dataset_input, np.ndarray):
            if dims is None:
                # handle single greyscale or rgb image:
                dataset_clean = PlaceDataset([dataset_input])
                # INPUTS: type list of np.ndarray greyscale or rgb images
            else:
                # handle compressed npz input:
                dataset_clean = PlaceDataset(dataset_input, dims)
                # INPUTS: type np.ndarray of np.ndarray flattened images
        elif isinstance(dataset_input, list): # np.ndarrays, PIL images, or strs:
            if any([isinstance(dataset_input[0], i) for i in [str, Image.Image, np.ndarray]]):
                dataset_clean = PlaceDataset(dataset_input)
                # INPUTS: type list of strs, PIL Image.Images, or np.ndarray images
            else:
                raise DescriptorContainerError('Bad input type.')
        else:
            raise DescriptorContainerError('Bad input type.')
        return dataset_clean

    def get_feat(self, dataset_input, dims=None, use_tqdm=True, save_dir=None):
        '''
        Perform feature extraction
        Accepts:
        Strings for path to a single image, or a directory containing other images
        list of np.ndarrays or PIL images or image paths
        A 2D np.ndarray (assumed single image)
        A 3D np.ndarray (first index is assumed to specify individual images)
        A 2D np.ndarray of flattened images (first index is assumed to specify individual images)
            (required dims to be specified)
        a PIL image
        '''
        dataset_clean = self.clean_data_input(dataset_input, dims)

        feats = []
        iteration_obj = dataset_clean
        if use_tqdm:
            iteration_obj = tqdm(dataset_clean)
        for (input_data, _) in iteration_obj:
            self.net.blobs['data'].data[...] = self.transformer.preprocess('data', input_data)
            self.net.forward()

            feat = np.squeeze(self.net.blobs[self.target_layer].data).flatten()
            feats.append(feat)
        del dataset_clean
        if len(feats) == 1:
            db_feat = feats[0]
        else:
            db_feat = np.array(feats)
        if not save_dir is None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            output_global_features_filename = os.path.join(save_dir, 'NetVLAD_feats.npy')
            np.save(output_global_features_filename, db_feat)
        return db_feat
