import numpy as np
from PIL import Image

from common.image import trans_square, clahe_detect, clahe_crop


def test_trans_square_makes_square():
    im = Image.fromarray((np.random.rand(20, 30, 3) * 255).astype("uint8"), "RGB")
    out = trans_square(im)
    assert out.size == (30, 30)


def test_trans_square_keeps_square_unchanged():
    im = Image.fromarray((np.random.rand(30, 30, 3) * 255).astype("uint8"), "RGB")
    assert trans_square(im).size == (30, 30)


def test_clahe_keeps_shape_and_dtype():
    gray = (np.random.rand(50, 40) * 255).astype("uint8")
    for fn in (clahe_detect, clahe_crop):
        out = fn(gray)
        assert out.shape == gray.shape
        assert out.dtype == np.uint8
