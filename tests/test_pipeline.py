import numpy as np
import pytest
import torch

from inference import pipeline


class _FakeDetector:
    def __init__(self, boxes):
        self._boxes = boxes

    def detect(self, img_path, conf=0.7):
        return np.zeros((100, 100, 3), dtype=np.uint8), self._boxes


def test_predict_raises_on_wrong_box_count(tmp_path):
    fake = _FakeDetector(torch.zeros((5, 6)))
    with pytest.raises(ValueError) as e:
        pipeline.predict_bone_age("x.png", "boy", fake, {})
    assert "预期 21" in str(e.value)


def test_predict_full_path_with_stub_classifier(tmp_path, monkeypatch):
    rows = []
    for x in (11, 12, 13, 14, 15, 16, 17):
        rows.append([x, 0, x + 5, 10, 0.9, 6])
    for x in (21, 22, 23):
        rows.append([x, 0, x + 5, 10, 0.9, 5])
    for x in (31, 32, 33, 34, 35):
        rows.append([x, 0, x + 5, 10, 0.9, 4])
    for x in (41, 42, 43):
        rows.append([x, 0, x + 5, 10, 0.9, 3])
    rows += [[51, 0, 56, 10, 0.9, 2], [61, 0, 66, 10, 0.9, 1], [71, 0, 76, 10, 0.9, 0]]
    fake = _FakeDetector(torch.tensor(rows, dtype=torch.float32))

    # 每个分类模型恒返回等级 0
    monkeypatch.setattr(pipeline, "classify_crop", lambda model, crop: 0)
    report = pipeline.predict_bone_age("x.png", "boy", fake, {name: object() for name in ("MCP", "MCPFirst", "DIP", "DIPFirst", "PIP", "PIPFirst", "MIP", "Radius", "Ulna")})

    assert "骨龄约为" in report
    assert "总得分" in report
