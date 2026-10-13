from pathlib import Path

from data_prepare.split_dataset import split_dataset


def test_split_dataset_creates_expected_layout(tmp_path):
    src_img = tmp_path / "JPEGImages"
    src_lbl = tmp_path / "labels"
    src_img.mkdir()
    src_lbl.mkdir()
    for i in range(10):
        (src_img / "im{}.png".format(i)).write_bytes(b"x")
        (src_lbl / "im{}.txt".format(i)).write_text("0 0.5 0.5 0.1 0.1\n")

    out = tmp_path / "out"
    counts = split_dataset(str(src_img), str(src_lbl), str(out), seed=0)

    assert counts["train"] + counts["val"] + counts["test"] == 10
    for split in ("train", "val", "test"):
        assert (out / "images" / split).is_dir()
        assert (out / "labels" / split).is_dir()
    total_img = sum(len(list((out / "images" / s).glob("*.png"))) for s in counts)
    total_lbl = sum(len(list((out / "labels" / s).glob("*.txt"))) for s in counts)
    assert total_img == 10
    assert total_lbl == 10
