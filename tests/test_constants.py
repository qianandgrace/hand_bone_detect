from common import constants


def test_detect_classes_has_7():
    assert len(constants.DETECT_CLASSES) == 7


def test_category_has_13_joints():
    assert len(constants.CATEGORY) == 13


def test_every_category_maps_to_a_model():
    for name in constants.CATEGORY:
        assert name in constants.ARTHROSIS
        model_name, num_cls = constants.ARTHROSIS[name]
        assert isinstance(model_name, str)
        assert num_cls > 1


def test_score_covers_all_categories_for_both_sexes():
    for sex in ("boy", "girl"):
        for name in constants.CATEGORY:
            assert name in constants.SCORE[sex]
            assert len(constants.SCORE[sex][name]) == constants.ARTHROSIS[name][1]
