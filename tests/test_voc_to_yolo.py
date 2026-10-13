from data_prepare.voc_to_yolo import voc_to_yolo


def _write_xml(path, name="Radius", xmin=10, ymin=20, xmax=30, ymax=40, w=100, h=200):
    path.write_text(
        "<annotation><size><width>{w}</width><height>{h}</height></size>"
        "<object><name>{name}</name><difficult>0</difficult>"
        "<bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin>"
        "<xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox></object></annotation>".format(
            w=w, h=h, name=name, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax
        ),
        encoding="utf-8",
    )


def test_voc_to_yolo_converts_one_box(tmp_path):
    xml_dir = tmp_path / "Annotations"
    xml_dir.mkdir()
    out_dir = tmp_path / "labels"
    _write_xml(xml_dir / "a.xml")

    n = voc_to_yolo(str(xml_dir), str(out_dir), ["Radius", "Ulna"])

    assert n == 1
    content = (out_dir / "a.txt").read_text().strip()
    assert content == "0 0.19 0.145 0.2 0.1"


def test_voc_to_yolo_skips_unknown_class(tmp_path):
    xml_dir = tmp_path / "Annotations"
    xml_dir.mkdir()
    out_dir = tmp_path / "labels"
    _write_xml(xml_dir / "b.xml", name="NotABone")

    voc_to_yolo(str(xml_dir), str(out_dir), ["Radius", "Ulna"])

    assert (out_dir / "b.txt").read_text() == ""
