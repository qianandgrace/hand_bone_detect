import numpy as np
from PIL import Image

from train.classify.dataset import My_data


def test_my_data_item_shape_and_label(tmp_path):
    img = tmp_path / "a.png"
    Image.fromarray((np.random.rand(40, 60) * 255).astype("uint8"), "L").save(img)
    (tmp_path / "train.txt").write_text("{} 2\n".format(img), encoding="utf-8")
    (tmp_path / "val.txt").write_text("", encoding="utf-8")

    ds = My_data(str(tmp_path), "train")
    x, y = ds[0]

    assert x.shape == (1, 224, 224)
    assert int(y) == 2


def test_my_data_empty_val(tmp_path):
    (tmp_path / "train.txt").write_text("", encoding="utf-8")
    (tmp_path / "val.txt").write_text("", encoding="utf-8")
    assert len(My_data(str(tmp_path), "val")) == 0
