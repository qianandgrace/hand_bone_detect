import yaml

from common.constants import DETECT_CLASSES


def test_config_names_match_detect_classes():
    with open("configs/bone_detect.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    assert [names[i] for i in range(len(names))] == DETECT_CLASSES
    for key in ("path", "train", "val"):
        assert key in cfg
