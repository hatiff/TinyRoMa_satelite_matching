import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as tfm
import warnings
from pathlib import Path
from typing import Tuple

from match.utils import to_normalized_coords, to_px_coords, to_numpy

class BaseMatcher(torch.nn.Module):

    # OpenCV default ransac params
    DEFAULT_RANSAC_ITERS = 2000
    DEFAULT_RANSAC_CONF = 0.99
    DEFAULT_REPROJ_THRESH = 3

    def __init__(self, device="cpu", **kwargs):
        super().__init__()
        self.usac = True
        self.device = device

        self.ransac_iters = kwargs.get("ransac_iters", BaseMatcher.DEFAULT_RANSAC_ITERS)
        self.ransac_conf = kwargs.get("ransac_conf", BaseMatcher.DEFAULT_RANSAC_CONF)
        self.ransac_reproj_thresh = kwargs.get("ransac_thresh", BaseMatcher.DEFAULT_REPROJ_THRESH)


    @staticmethod
    def load_image(path: str | Path, resize: int | Tuple = None, rot_angle: float = 0) -> torch.Tensor:
        if isinstance(resize, int):
            resize = (resize, resize)
        img = tfm.ToTensor()(Image.open(path).convert("RGB"))
        if resize is not None:
            img = tfm.Resize(resize, antialias=True)(img)
        img = tfm.functional.rotate(img, rot_angle)
        return img

    def rescale_coords(
        self,
        pts: np.ndarray | torch.Tensor,
        h_orig: int,
        w_orig: int,
        h_new: int,
        w_new: int,
    ) -> np.ndarray:
        """Keypoints rescailing"""
        return to_px_coords(to_normalized_coords(pts, h_new, w_new), h_orig, w_orig)

    @staticmethod
    def use_usac(
        points1: np.ndarray | torch.Tensor,
        points2: np.ndarray | torch.Tensor,
        reproj_thresh: int = DEFAULT_REPROJ_THRESH,
        num_iters: int = DEFAULT_RANSAC_ITERS,
        ransac_conf: float = DEFAULT_RANSAC_CONF,
    ):
        assert points1.shape == points2.shape
        assert points1.shape[1] == 2
        points1, points2 = to_numpy(points1), to_numpy(points2)

        H, inliers_mask = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, reproj_thresh, ransac_conf, num_iters)
        assert inliers_mask.shape[1] == 1
        inliers_mask = inliers_mask[:, 0]
        return H, inliers_mask.astype(bool)

    def process_matches(
        self, match_kpts0: np.ndarray, match_kpts1: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Processing matches with RANSAC (USAC)."""

        if len(match_kpts0) < 4:
            return None, match_kpts0, match_kpts1
        
        if self.usac:
            H, inliers_mask = self.use_usac(
                match_kpts0,
                match_kpts1,
                self.ransac_reproj_thresh,
                self.ransac_iters,
                self.ransac_conf,
            )
        inlier_kpts0 = match_kpts0[inliers_mask]
        inlier_kpts1 = match_kpts1[inliers_mask]

        return H, inlier_kpts0, inlier_kpts1

    def preprocess(self, img: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Image preprocessing""" 
        _, h, w = img.shape
        orig_shape = h, w
        return img, orig_shape

    @torch.inference_mode()
    def forward(self, img0: torch.Tensor | str | Path, img1: torch.Tensor | str | Path) -> dict:
        
        # Take as input a pair of images (not a batch)
        if isinstance(img0, (str, Path)):
            img0 = BaseMatcher.load_image(img0)
        if isinstance(img1, (str, Path)):
            img1 = BaseMatcher.load_image(img1)

        assert isinstance(img0, torch.Tensor)
        assert isinstance(img1, torch.Tensor)

        img0 = img0.to(self.device)
        img1 = img1.to(self.device)

        # self._forward() is implemented by the children modules
        matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1 = self._forward(img0, img1)

        matched_kpts0, matched_kpts1 = to_numpy(matched_kpts0), to_numpy(matched_kpts1)
        H, inlier_kpts0, inlier_kpts1 = self.process_matches(matched_kpts0, matched_kpts1)

        return {
            "num_inliers": len(inlier_kpts0),
            "H": H,
            "all_kpts0": to_numpy(all_kpts0),
            "all_kpts1": to_numpy(all_kpts1),
            "all_desc0": to_numpy(all_desc0),
            "all_desc1": to_numpy(all_desc1),
            "matched_kpts0": matched_kpts0,
            "matched_kpts1": matched_kpts1,
            "inlier_kpts0": inlier_kpts0,
            "inlier_kpts1": inlier_kpts1,
        }

    def extract(self, img: str | Path | torch.Tensor) -> dict:
        # Take as input a pair of images (not a batch)
        if isinstance(img, (str, Path)):
            img = BaseMatcher.load_image(img)
        assert isinstance(img, torch.Tensor)
        img = img.to(self.device)
        matched_kpts0, _, all_kpts0, _, all_desc0, _ = self._forward(img, img)
        return {"all_kpts0": to_numpy(all_kpts0), "all_desc0": to_numpy(all_desc0)}


