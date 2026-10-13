import cv2
import numpy as np
from PIL import Image


def trans_square(image):
    """把 PIL 图片等比补黑边成正方形。"""
    img = image.convert("RGB")
    img = np.array(img, dtype=np.uint8)
    img_h, img_w, img_c = img.shape
    if img_h != img_w:
        long_side = max(img_w, img_h)
        short_side = min(img_w, img_h)
        loc = abs(img_w - img_h) // 2
        img = img.transpose((1, 0, 2)) if img_w < img_h else img
        background = np.zeros((long_side, long_side, img_c), dtype=np.uint8)
        background[loc:loc + short_side] = img[...]
        img = background.transpose((1, 0, 2)) if img_w < img_h else background
    return Image.fromarray(img, "RGB")


def clahe_detect(gray):
    """整张 X 光检测前的 CLAHE（参数与推理原有实现一致）。"""
    return cv2.createCLAHE(tileGridSize=(5, 5)).apply(gray)


def clahe_crop(gray):
    """关节裁剪图分类前的 CLAHE（参数与训练/推理原有实现一致）。"""
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3)).apply(gray)
