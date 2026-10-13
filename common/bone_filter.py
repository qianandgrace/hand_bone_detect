import torch


def _select(boxes, cls, flags):
    # 取指定类别的框，按 x1 升序排序后，选指定下标的框
    boxes = boxes[torch.where(boxes[:, 5] == cls)]
    order = boxes[:, 0].argsort()
    return boxes[order][flags]


def filter_joints(boxes):
    """把检测输出的 21 个框筛选排序成 13 个目标关节。

    boxes: Tensor[N, 6]，列为 [x1, y1, x2, y2, conf, cls]
    返回: Tensor[13, 6]，顺序与 common.constants.CATEGORY 一致
    """
    if boxes.shape[0] != 21:
        raise ValueError(
            "检测到 {} 个框，预期 21，请检查检测模型/置信度阈值".format(boxes.shape[0])
        )
    distal = _select(boxes, 6, [0, 2, 4])      # DistalPhalanx
    middle = _select(boxes, 5, [0, 2])         # MiddlePhalanx
    proximal = _select(boxes, 4, [0, 2, 4])    # ProximalPhalanx
    mcp = _select(boxes, 3, [0, 2])            # MCP
    mcp_first = _select(boxes, 2, [0])         # MCPFirst
    ulna = _select(boxes, 1, [0])              # Ulna
    radius = _select(boxes, 0, [0])            # Radius
    return torch.cat([distal, middle, proximal, mcp, mcp_first, ulna, radius], 0)
