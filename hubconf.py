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

RESNET50_URL = "https://www.dropbox.com/s/wlaey1rk8id2i5t/retinaface_resnet50.pth?dl=1"


def retinaface_resnet50(pretrained=True, **kwargs):

    model = RetinaFace(cfg=CFG_RESNET50, phase="test")
    model.eval()

    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(RESNET50_URL, progress=False)
        )

    return model
