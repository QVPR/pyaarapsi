#!/usr/bin/env python3
'''
Tools for LightGlue. Requires a separate LightGlue/SuperPoint installation
'''
import warnings
from typing import Union, Tuple, Callable
from cv2 import findEssentialMat as cv_findEssentialMat, recoverPose as cv_recoverPose \
    #pylint: disable=E0611
import numpy as np
from numpy.typing import NDArray
try:
    from lightglue import LightGlue, SuperPoint # type: ignore
    from lightglue.utils import rbd # type: ignore
except ImportError as e:
    raise ImportError("Could not find LightGlue installation.") from e
import torch
from torch import tensor as t_tensor #pylint: disable=E0611
from pyaarapsi.core.transforms import rotation_matrix_to_euler_angles

class LightGlueMatcher:
    '''
    Wrapper class to handle LightGlue correspondence matching
    '''
    def __init__(self, img_dims: Tuple[int,int] = (64,64), rgb: bool = False):
        torch.set_grad_enabled(False)
        self.device     = torch.device("cuda" if torch.cuda.is_available() \
                                       else "cpu") # 'mps', 'cpu'
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        self.matcher: Callable = LightGlue(features="superpoint").eval().to(self.device)
        self.img_dims = img_dims
        self.rgb = rgb
    #
    def img2torch(self, img: NDArray, reshape=True) -> torch.Tensor:
        '''
        Convert an image to a torch tensor
        '''
        if reshape:
            img = np.reshape(img, (self.img_dims[1], self.img_dims[0]))
        if not self.rgb:
            img = img[None]
        return t_tensor((img - np.min(img)) / (np.max(img) - np.min(img)), dtype=torch.float)
    #
    def match_images(self, input_0: Union[NDArray, torch.Tensor],
                     input_1: Union[NDArray, torch.Tensor]) -> Tuple[NDArray, NDArray]:
        '''
        Calculate correspondences, helper for if inputs are NDArrays or Tensors
        '''
        if isinstance(input_0, np.ndarray):
            tensor_0 = self.img2torch(input_0)
        else:
            tensor_0 = input_0
        if isinstance(input_1, np.ndarray):
            tensor_1 = self.img2torch(input_1)
        else:
            tensor_1 = input_1
        return self.match_tensors(tensor_0, tensor_1)
    #
    def match_tensors(self, tensor_0: torch.Tensor, tensor_1: torch.Tensor
                      ) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Calculate correspondences
        '''
        sp_feats_0 = self.extractor.extract(tensor_0.to(self.device))
        sp_feats_1 = self.extractor.extract(tensor_1.to(self.device))

        lg_matches_raw_dict = self.matcher({"image0": sp_feats_0, "image1": sp_feats_1}) \
            #pylint: disable=E1102

        lg_feats_0, lg_feats_1, lg_matches_dict = [
            rbd(x) for x in [sp_feats_0, sp_feats_1, lg_matches_raw_dict]
        ]  # remove batch dimension

        key_pts_0, key_pts_1, lg_matches = lg_feats_0["keypoints"], lg_feats_1["keypoints"], \
                                                lg_matches_dict["matches"]
        matched_key_pts_0, matched_key_pts_1 = key_pts_0[lg_matches[..., 0]].cpu().numpy(), \
                                                    key_pts_1[lg_matches[..., 1]].cpu().numpy()

        return matched_key_pts_0, matched_key_pts_1
    #
    def set_rgb(self, rgb: bool):
        '''
        Set colour enabled/disabled
        '''
        self.rgb = rgb
    #
    def set_img_dims(self, img_dims: Tuple[int,int]):
        '''
        Set image dimensions
        '''
        self.img_dims = img_dims

class PoseEstimator(LightGlueMatcher):
    '''
    Wrapper on LightGlueMatcher to provide pose estimation functionality, separately.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    #
    def estimate_pose_between_images(self, input_0: Union[np.ndarray, torch.Tensor],
                                  input_1: Union[np.ndarray, torch.Tensor]
                                  ) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Use cv2 to perform pose matching
        '''
        matched_key_pts_0, matched_key_pts_1 = self.match_images(input_0, input_1)
        essential_matrix, _ = cv_findEssentialMat(matched_key_pts_0, matched_key_pts_1)
        _, cv_rotation_matrix, cv_translation_matrix, _ = \
            cv_recoverPose(essential_matrix, matched_key_pts_0, matched_key_pts_1)
        warnings.warn("[PoseEstimator::estimate_pose_between_images] Fairly sure this code is "
                      "wrong, beware. I don't think cv_recoverPose returns an Euler angle matrix.")
        rotation_roll_pitch_yaw = rotation_matrix_to_euler_angles(cv_rotation_matrix)
        translation_scaled = cv_translation_matrix.flatten()
        return rotation_roll_pitch_yaw, translation_scaled
