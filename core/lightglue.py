import cv2
import numpy as np
from pathlib import Path
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import torch
from pyaarapsi.core.transforms import rotationMatrixToEulerAngles
from typing import Union, Tuple

class LightGlueMatcher:
    def __init__(self, img_dims: Tuple[int,int] = (64,64), rgb: bool = False):
        torch.set_grad_enabled(False)
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
        self.extractor  = SuperPoint(max_num_keypoints=2048).eval().to(self.device)  # load the extractor
        self.matcher    = LightGlue(features="superpoint").eval().to(self.device)
        self.img_dims   = img_dims
        self.rgb        = rgb

    def img2torch(self, img: np.ndarray, reshape=True) -> torch.Tensor:
        if reshape:
            img = np.reshape(img, (self.img_dims[1], self.img_dims[0]))
        if not self.rgb:
            img = img[None]
        return torch.tensor((img - np.min(img)) / (np.max(img) - np.min(img)), dtype=torch.float)

    def match_images(self, input_0: Union[np.ndarray, torch.Tensor], input_1: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(input_0, np.ndarray):
            tensor_0 = self.img2torch(input_0)
        else:
            tensor_0 = input_0

        if isinstance(input_1, np.ndarray):
            tensor_1 = self.img2torch(input_1)
        else:
            tensor_1 = input_1

        return self.match_tensors(tensor_0, tensor_1)
    
    def match_tensors(self, tensor_0: torch.Tensor, tensor_1: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        sp_feats_0 = self.extractor.extract(tensor_0.to(self.device))
        sp_feats_1 = self.extractor.extract(tensor_1.to(self.device))

        lg_matches_raw_dict = self.matcher({"image0": sp_feats_0, "image1": sp_feats_1})

        lg_feats_0, lg_feats_1, lg_matches_dict = [
            rbd(x) for x in [sp_feats_0, sp_feats_1, lg_matches_raw_dict]
        ]  # remove batch dimension

        key_pts_0, key_pts_1, lg_matches        = lg_feats_0["keypoints"], lg_feats_1["keypoints"], lg_matches_dict["matches"]
        matched_key_pts_0, matched_key_pts_1    = key_pts_0[lg_matches[..., 0]].cpu().numpy(), key_pts_1[lg_matches[..., 1]].cpu().numpy()

        return matched_key_pts_0, matched_key_pts_1
    
    def set_rgb(self, rgb: bool):
        self.rgb = rgb
    
    def set_img_dims(self, img_dims: Tuple[int,int]):
        self.img_dims = img_dims

class PoseEstimator(LightGlueMatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def estimatePoseBetweenImages(self, input_0: Union[np.ndarray, torch.Tensor], input_1: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        matched_key_pts_0, matched_key_pts_1 = self.match_images(input_0, input_1)

        _E, _E_mask = cv2.findEssentialMat(matched_key_pts_0, matched_key_pts_1)
        _points, _R, _t, _rp_mask = cv2.recoverPose(_E, matched_key_pts_0, matched_key_pts_1)

        rotation_roll_pitch_yaw = rotationMatrixToEulerAngles(_R)
        translation_scaled = _t.flatten()
        return rotation_roll_pitch_yaw, translation_scaled