import os

import cv2
from ultralytics import YOLO

from common.image import clahe_detect


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DEFAULT_WEIGHTS = os.path.join(_repo_root(), "models", "yolov5su.pt")


def load_detector(weights=None):
    weights = weights or DEFAULT_WEIGHTS
    if not os.path.exists(weights):
        raise FileNotFoundError("检测权重不存在: {}，请先训练或下载".format(weights))
    return YOLO(weights)


def detect(model, img_path, conf=0.7):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = model.predict(clahe_detect(gray), conf=conf, verbose=False)[0]
    return img, result.boxes.data


class Detector:
    """检测器适配：把「模型」包成 pipeline 期望的 `.detect(img_path, conf)` 接口。"""

    def __init__(self, weights=None):
        self.model = load_detector(weights)

    def detect(self, img_path, conf=0.7):
        return detect(self.model, img_path, conf)
