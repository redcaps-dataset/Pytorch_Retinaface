dependencies = ["torch"]

import torch

from models.retinaface import RetinaFace


CFG_RESNET50 = {
    "name": "Resnet50",
    "pretrain": False,
    "return_layers": {"layer2": 1, "layer3": 2, "layer4": 3},
    "in_channel": 256,
    "out_channel": 256
}


def retinaface_resnet50(landmark_head: bool = True, **kwargs):

    model = RetinaFace(cfg=CFG_RESNET50, phase="test")
    model.eval()

    if not landmark_head:
        model.LandmarkHead = torch.nn.Identity()

    return model
