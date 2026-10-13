from pathlib import Path

from data_prepare.prepare_classify import prepare_classify


def _make_dataset(tmp_path):
    root = tmp_path / "arthrosis"
    joint = root / "MCP"
    for grade in ("1", "2"):
        d = joint / grade
        d.mkdir(parents=True)
        for i in range(5):
            (d / "g{}_{}.png".format(grade, i)).write_bytes(b"x")
    return str(root)


def test_prepare_classify_writes_txt(tmp_path):
    root = _make_dataset(tmp_path)
    stats = prepare_classify(root, train_ratio=0.8, seed=0)

    joint_dir = Path(root) / "MCP"
    train_lines = (joint_dir / "train.txt").read_text().strip().splitlines()
    val_lines = (joint_dir / "val.txt").read_text().strip().splitlines()

    assert len(train_lines) + len(val_lines) == 10
    assert stats["MCP"]["train"] == 8
    assert stats["MCP"]["val"] == 2
    # 标签 = 等级目录名 - 1
    labels = {int(line.rsplit(" ", 1)[1]) for line in train_lines + val_lines}
    assert labels == {0, 1}
