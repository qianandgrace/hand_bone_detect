import torch
from PIL import Image
from torch import nn

from inference import classifier


class _RecordingModule(nn.Module):
    """替身模块：记录收到输入所在设备，返回同设备的固定输出。"""

    def __init__(self):
        super().__init__()
        self.seen_device = None

    def forward(self, x):
        self.seen_device = x.device
        return torch.zeros(x.shape[0], 2, device=x.device)


def test_classify_crop_moves_input_to_device():
    model = _RecordingModule()
    img = Image.new("L", (224, 224), 128)

    classifier.classify_crop(model, img)

    assert model.seen_device is not None
    assert model.seen_device.type == torch.device(classifier.DEVICE).type
