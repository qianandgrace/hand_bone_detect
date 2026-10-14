import numpy as np
import torch

from inference import detector


class _StubPredictor:
    """鸭子类型替身：记录输入，返回固定的 ultralytics 风格结果。"""

    def __init__(self):
        self.seen_conf = None
        self.seen_input_shape = None

    def predict(self, image, conf=0.25, verbose=False):
        self.seen_conf = conf
        self.seen_input_shape = image.shape

        class _Boxes:
            data = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.9, 0.0]])

        class _Result:
            boxes = _Boxes()

        return [_Result()]


def test_detect_returns_image_and_boxes(tmp_path):
    import cv2

    img_path = tmp_path / "x.png"
    cv2.imwrite(str(img_path), np.zeros((30, 40, 3), dtype=np.uint8))

    stub = _StubPredictor()
    img, boxes = detector.detect(stub, str(img_path), conf=0.5)

    assert img.shape == (30, 40, 3)
    assert boxes.shape == (1, 6)
    assert stub.seen_conf == 0.5
    assert stub.seen_input_shape == (30, 40)   # 送入的是灰度图


def test_load_detector_missing_weights_raises(tmp_path):
    missing = str(tmp_path / "nope.pt")
    try:
        detector.load_detector(missing)
    except FileNotFoundError as exc:
        assert "请先训练或下载" in str(exc)
    else:
        raise AssertionError("应抛出 FileNotFoundError")


def test_detector_delegates_to_module_detect(monkeypatch, tmp_path):
    import cv2

    stub = _StubPredictor()
    monkeypatch.setattr(detector, "load_detector", lambda weights=None: stub)

    img_path = tmp_path / "y.png"
    cv2.imwrite(str(img_path), np.zeros((30, 40, 3), dtype=np.uint8))

    det = detector.Detector()
    assert det.model is stub

    img, boxes = det.detect(str(img_path), conf=0.4)

    assert img.shape == (30, 40, 3)
    assert boxes.shape == (1, 6)
    assert stub.seen_conf == 0.4
    assert stub.seen_input_shape == (30, 40)
