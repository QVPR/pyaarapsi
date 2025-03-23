#!/usr/bin/env python3
'''
NetVLAD Container for feature extraction
'''
import configparser
import os

from tqdm.auto import tqdm
import torch
from torch import zeros as t_zeros #pylint: disable=E0611
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image

from pyaarapsi.vpr.classes.descriptors.lib.netvlad_lib import get_backend, get_model, \
    get_pca_encoding
from pyaarapsi.vpr.classes.descriptors.generic import GenericPlaceDataset, \
    PlaceDatasetError, DescriptorContainerError, DescriptorContainer
from pyaarapsi.vpr.classes.download_models import download_netvlad_models, MODELS_DIR

class PlaceDataset(GenericPlaceDataset):
    '''
    NetVLAD dataset handler
    '''
    def __init__(self, image_data, transform, dims=None):
        super().__init__()

        self.images = image_data
        self.transform = transform
        self.dims = dims
        self.len = None
        self.getitem = None

        if isinstance(self.images, list):
            # either contains PIL Images, str paths to images, or np.ndarray images
            self.len = lambda: len(self.images)
            if isinstance(self.images[0], Image.Image): # type list of PIL Images
                self.getitem = lambda index: self.images[index]
            elif isinstance(self.images[0], str): # type list of strs
                self.getitem = lambda index: Image.open(self.images[index])
            elif isinstance(self.images[0], np.ndarray): # type list of np.ndarray images
                if len(self.images[0].shape) == 2: # grayscale
                    self.getitem = \
                        lambda index: Image.fromarray(
                            np.dstack((self.images[index],)*3).astype(np.uint8))
                else:
                    self.getitem = \
                        lambda index: Image.fromarray(
                            self.images[index].astype(np.uint8))
            else:
                raise PlaceDatasetError("Input of type list but contains elements of unknown "
                                        f"type. Type: {str(type(self.images[0]))}")
        elif isinstance(self.images, np.ndarray): # type np.ndarray of np.ndarray flattened images
            self.len = lambda: self.images.shape[0]
            self.getitem = lambda index: \
                Image.fromarray(np.dstack((np.reshape(self.images[index], self.dims),)*3) \
                                    .astype(np.uint8))
        else:
            raise PlaceDatasetError("Input not of type list or np.ndarray. "
                                    f"Type: {str(type(self.images))}")

    def __getitem__(self, index):
        '''
        Get items
        '''
        return self.transform(self.getitem(index)), index

    def __len__(self):
        '''
        Length of dataset
        '''
        return self.len()

    def __del__(self):
        '''
        Clean-up
        '''
        if self.len is not None:
            del self.len
        if self.getitem is not None:
            del self.getitem
        del self.images
        self.transform = None # comes from NetVLAD_Container
        del self.dims

class NetVLADContainer(DescriptorContainer):
    '''
    Feature extraction for NetVLAD
    '''
    def __init__(self, logger=print, cuda=False, ngpus=0,
                 imw=640, imh=480, batchsize=5, cachebatchsize=5, num_pcs=4096,
                 threads=0, resumepath='/mapillary_WPCA',
                 load=True, prep=True):
        #
        self.cuda           = cuda
        self.ngpus          = ngpus
        self.logger         = logger
        self.imw            = imw
        self.imh            = imh
        self.batchsize      = batchsize
        self.cachebatchsize = cachebatchsize
        self.num_pcs        = num_pcs
        self.threads        = threads
        self.resumepath     = resumepath
        self.transform      = self.input_transform()
        self.loaded         = False
        self.prepped        = False
        self.config         = None
        self.device         = None
        self.model          = None
        if load:
            self.load()
        if prep:
            self.prep()

    def is_ready(self):
        '''
        Whether model can be used right now
        '''
        return self.prepped

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

    def load(self):
        '''
        Load model data
        '''
        self.config = configparser.ConfigParser()
        self.config['feature_extract'] = {'batchsize': self.batchsize,
                                            'cachebatchsize': self.cachebatchsize,
                                            'imageresizew': self.imw, 'imageresizeh': self.imh}
        self.config['global_params'] = {'pooling': 'netvlad', 'resumepath': self.resumepath,
                                            'threads': self.threads, 'num_pcs': self.num_pcs,
                                            'ngpu': self.ngpus}

        if self.cuda and not torch.cuda.is_available():
            raise DescriptorContainerError("No GPU found")
        self.device = torch.device("cuda" if self.cuda else "cpu")

        # must resume to do extraction
        if int(self.config['global_params']['num_pcs']) > 0:
            resume_ckpt = self.config['global_params']['resumePath'] \
                            + self.config['global_params']['num_pcs'] + '.pth.tar'
        else:
            resume_ckpt = self.config['global_params']['resumePath'] + '.pth.tar'

        file_path = MODELS_DIR + resume_ckpt
        if not os.path.isfile(file_path):
            download_netvlad_models()

        if os.path.isfile(file_path):
            self.logger(f"=> Trying to load checkpoint '{file_path}'")
            checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage, weights_only=False)
            if bool(self.num_pcs):
                assert checkpoint['state_dict']['WPCA.0.bias'].shape[0] \
                    == int(self.config['global_params']['num_pcs'])
            self.config['global_params']['num_clusters'] = \
                str(checkpoint['state_dict']['pool.centroids'].shape[0])
            encoder_dim, encoder = get_backend()
            self.model = get_model(encoder, encoder_dim, self.config['global_params'],
                                   append_pca_layer=bool(self.num_pcs))
            del encoder, encoder_dim
            self.model.load_state_dict(checkpoint['state_dict'])
            del checkpoint
            if int(self.config['global_params']['ngpu']) > 1 and torch.cuda.device_count() > 1:
                self.model.encoder = torch.nn.DataParallel(self.model.encoder)
                self.model.pool = torch.nn.DataParallel(self.model.pool)
            self.model = self.model.to(self.device)
            self.logger(f"=> Successfully loaded checkpoint '{file_path}'")
        else:
            raise FileNotFoundError(f"=> no checkpoint found at '{file_path}'")
        self.model.eval()
        self.loaded = True

    def __del__(self):
        '''
        Destroy all data, objects
        '''
        del self.cuda
        del self.ngpus
        del self.logger
        del self.imw
        del self.imh
        del self.batchsize
        del self.cachebatchsize
        del self.num_pcs
        del self.threads
        del self.resumepath
        del self.transform
        del self.loaded
        del self.prepped
        if self.config is not None:
            del self.config
        if self.device is not None:
            del self.device
        if self.model is not None:
            del self.model

    def prep(self):
        '''
        Any prepwork, such as passing a default object through the container
        '''
        torch.cuda.empty_cache()
        torch.nn.functional.conv2d(t_zeros(32, 32, 32, 32, device=self.device), \
                            t_zeros(32, 32, 32, 32, device=self.device)) # pylint: disable=E1102

        input_data = self.transform(Image.fromarray(np.zeros((1,1,3), dtype=np.uint8)))
        with torch.no_grad():
            input_data      = input_data.unsqueeze(dim=0).to(self.device)
            image_encoding  = self.model.encoder(input_data)
            vlad_global     = self.model.pool(image_encoding)
            vlad_global_pca = get_pca_encoding(self.model, vlad_global)
        del input_data, image_encoding, vlad_global, vlad_global_pca
        torch.cuda.empty_cache()
        self.prepped = True

    def input_transform(self):
        '''
        Transform function
        '''
        return transforms.Compose([
            transforms.Resize((self.imh, self.imw)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    def clean_data_input(self, dataset_input, dims):
        '''
        Process input
        '''
        if isinstance(dataset_input, str):
            # either a single image path or a set (or a bad path has been provided)
            try:
                img = Image.open(dataset_input)
                if img is None:
                    raise DescriptorContainerError('Image path is invalid')
                dataset_clean = PlaceDataset([dataset_input], self.transform) # handle single image
                # INPUTS: type list of strs
            except (PlaceDatasetError, DescriptorContainerError):
                try:
                    ref_filenames = [os.path.join(dataset_input, filename)
                                    for filename in sorted(os.listdir(dataset_input))
                                    if os.path.isfile(os.path.join(dataset_input, filename))]
                    # handle directory of images:
                    dataset_clean = PlaceDataset(ref_filenames, self.transform)
                    # INPUTS: type list of strs
                except Exception as e:
                    raise DescriptorContainerError("Bad path specified; input is not an image or "
                                            "directory path") from e
        elif isinstance(dataset_input, Image.Image):
            dataset_clean = PlaceDataset([dataset_input], self.transform) # handle PIL image
            # INPUTS: type list of PIL Images
        elif isinstance(dataset_input, np.ndarray):
            if dims is None:
                # handle single greyscale or rgb image:
                dataset_clean = PlaceDataset([dataset_input], self.transform)
                # INPUTS: type list of PIL Images
            else:
                # handle compressed npz input:
                dataset_clean = PlaceDataset(dataset_input, self.transform, dims)
                # INPUTS: type np.ndarray of np.ndarray flattened images
        elif isinstance(dataset_input, list): # np.ndarrays, PIL images, or strs:
            if any([isinstance(dataset_input[0], i) for i in [str, Image.Image, np.ndarray]]):
                dataset_clean = PlaceDataset(dataset_input, self.transform)
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
        data_loader = DataLoader(dataset = dataset_clean,
                        num_workers = int(self.config['global_params']['threads']),
                        batch_size = int(self.config['feature_extract']['cacheBatchSize']),
                        shuffle = False, pin_memory = self.cuda)
        with torch.no_grad():
            db_feat = np.empty((len(dataset_clean), int(self.config['global_params']['num_pcs'])),
                               dtype=np.float32)
            if use_tqdm:
                iteration_obj = tqdm(data_loader)
            else:
                iteration_obj = data_loader

            for (input_data, indices) in iteration_obj: # manage batches and threads
                indices_np              = indices.detach().numpy()
                input_data              = input_data.to(self.device)
                image_encoding          = self.model.encoder(input_data)
                vlad_global             = self.model.pool(image_encoding)
                vlad_global_pca         = get_pca_encoding(self.model, vlad_global)
                db_feat[indices_np, :]  = vlad_global_pca.detach().cpu().numpy()
                torch.cuda.empty_cache()
        del dataset_clean
        if not save_dir is None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            output_global_features_filename = os.path.join(save_dir, 'NetVLAD_feats.npy')
            np.save(output_global_features_filename, db_feat)
        return db_feat
