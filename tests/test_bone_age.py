import pytest

from common.bone_age import calc_bone_age, export_report


def test_calc_bone_age_known_values():
    assert calc_bone_age(500, "boy") == 12.59
    assert calc_bone_age(500, "girl") == 10.39
    assert calc_bone_age(1000, "boy") == 18.21


def test_calc_bone_age_rejects_bad_sex():
    with pytest.raises(ValueError):
        calc_bone_age(500, "cat")


def test_export_report_contains_total_and_age():
    results = {name: [3, 20] for name in (
        "MCPFirst", "MCPThird", "MCPFifth", "PIPFirst", "PIPThird", "PIPFifth",
        "MIPThird", "MIPFifth", "DIPFirst", "DIPThird", "DIPFifth", "Ulna", "Radius",
    )}
    text = export_report(results, 260, 11.5)
    assert "260" in text
    assert "11.5" in text
