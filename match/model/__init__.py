from .tiny_roma_model import TinyRoMa

import torch


weight_url = {
    "tiny_roma_v1": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/tiny_roma_v1_outdoor.pth",
    },

}



def tiny_roma(weights = None, freeze_xfeat=False, exact_softmax=False, xfeat = None):
    model = TinyRoMa(
        xfeat = xfeat,
        freeze_xfeat=freeze_xfeat, 
        exact_softmax=exact_softmax)
    if weights is not None:
        model.load_state_dict(weights)
    return model


def tinyroma(device, weights = None, xfeat = None):
    
    if weights is None:
        weights = torch.hub.load_state_dict_from_url(
            weight_url["tiny_roma_v1"]["outdoor"],
            map_location=device)
    if xfeat is None:
        xfeat = torch.hub.load(
            'verlab/accelerated_features', 
            'XFeat', 
            pretrained = True, 
            top_k = 4096).net
    return tiny_roma(weights = weights, xfeat = xfeat).to(device) 
    
