import torch
import pytest

from common.bone_filter import filter_joints


def _make_boxes():
    rows = []
    for x in (11, 12, 13, 14, 15, 16, 17):   # DistalPhalanx  cls=6 -> 选 11,13,15
        rows.append([x, 0, x + 5, 10, 0.9, 6])
    for x in (21, 22, 23):                    # MiddlePhalanx  cls=5 -> 选 21,23
        rows.append([x, 0, x + 5, 10, 0.9, 5])
    for x in (31, 32, 33, 34, 35):            # ProximalPhalanx cls=4 -> 选 31,33,35
        rows.append([x, 0, x + 5, 10, 0.9, 4])
    for x in (41, 42, 43):                    # MCP            cls=3 -> 选 41,43
        rows.append([x, 0, x + 5, 10, 0.9, 3])
    rows.append([51, 0, 56, 10, 0.9, 2])      # MCPFirst       cls=2
    rows.append([61, 0, 66, 10, 0.9, 1])      # Ulna           cls=1
    rows.append([71, 0, 76, 10, 0.9, 0])      # Radius         cls=0
    boxes = torch.tensor(rows, dtype=torch.float32)
    torch.manual_seed(0)
    return boxes[torch.randperm(boxes.shape[0])]   # 打乱，验证不依赖输入顺序


def test_filter_joints_shape_and_order():
    out = filter_joints(_make_boxes())
    assert out.shape == (13, 6)
    assert [int(v) for v in out[:, 0]] == [11, 13, 15, 21, 23, 31, 33, 35, 41, 43, 51, 61, 71]
    assert [int(v) for v in out[:, 5]] == [6, 6, 6, 5, 5, 4, 4, 4, 3, 3, 2, 1, 0]


def test_filter_joints_raises_on_wrong_count():
    with pytest.raises(ValueError):
        filter_joints(torch.zeros((20, 6)))
