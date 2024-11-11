import torch
import torchvision.transforms as tfm

from pathlib import Path
from .model import tinyroma
from .base_matcher import BaseMatcher


WEIGHTS_DIR = Path(__file__).parent.joinpath("model_weights")
WEIGHTS_DIR.mkdir(exist_ok=True)


class TinyRomaMatcher(BaseMatcher):

    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.roma_model = tinyroma(device=device)
        self.max_keypoints = max_num_keypoints
        self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.roma_model.train(False)

    def preprocess(self, img):
        return self.normalize(img).unsqueeze(0)

    def _forward(self, img0, img1):
        
        img0 = self.preprocess(img0)
        img1 = self.preprocess(img1)

        h0, w0 = img0.shape[-2:]
        h1, w1 = img1.shape[-2:]

        # batch = {"im_A": img0.to(self.device), "im_B": img1.to(self.device)}
        warp, certainty = self.roma_model.match(img0, img1, batched=False)

        matches, certainty = self.roma_model.sample(warp, certainty, num=self.max_keypoints)
        mkpts0, mkpts1 = self.roma_model.to_pixel_coordinates(matches, h0, w0, h1, w1)

        return mkpts0, mkpts1, None, None, None, None