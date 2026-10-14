import os

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models
from torchvision.transforms import Compose, InterpolationMode, Resize, ToTensor

from common.constants import ARTHROSIS
from common.image import clahe_crop, trans_square

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAMES = list(dict.fromkeys(v[0] for v in ARTHROSIS.values()))
NUM_CLASSES = {v[0]: v[1] for v in ARTHROSIS.values()}

data_transforms = Compose([
    Resize(size=(224, 224), interpolation=InterpolationMode.NEAREST),
    ToTensor(),
])


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_classifiers(base_dir=None):
    base_dir = base_dir or os.path.join(_repo_root(), "models", "classify")
    cls_models = {}
    for name in MODEL_NAMES:
        path = os.path.join(base_dir, "{}_best.pth".format(name))
        if not os.path.exists(path):
            raise FileNotFoundError("分类权重不存在: {}，请先训练".format(path))
        net = models.resnet18()
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        net.fc = nn.Linear(512, NUM_CLASSES[name])
        net.load_state_dict(torch.load(path, map_location=DEVICE))
        net = net.to(DEVICE)
        net.eval()
        cls_models[name] = net
    return cls_models


def classify_crop(model, pil_img):
    """对单个关节裁剪图分级，返回 0 起的等级。"""
    im = trans_square(pil_img).convert("L")
    im = clahe_crop(np.array(im))
    im = data_transforms(Image.fromarray(im))
    im = torch.unsqueeze(im, dim=0).to(DEVICE)
    with torch.no_grad():
        out = model(im)
    return int(out.argmax(dim=1))
