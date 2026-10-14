import numpy as np
import torch

from inference import predict


def test_run_reports_box_count_error(monkeypatch, capsys):
    class _FakeDetector:
        def __init__(self, weights=None):
            pass

        def detect(self, img_path, conf=0.7):
            return np.zeros((10, 10, 3), dtype=np.uint8), torch.zeros((7, 6))

    monkeypatch.setattr(predict, "Detector", _FakeDetector)
    monkeypatch.setattr(predict, "load_classifiers", lambda base_dir=None: {})

    code = predict._run("x.png", "boy", 0.7, None, None)

    assert code == 1
    assert "预期 21" in capsys.readouterr().err
