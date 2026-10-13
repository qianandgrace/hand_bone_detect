# 骨龄检测项目重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把骨龄检测仓库重整为「共享核心包 + 数据处理 / 训练 / 推理」三大模块，并删除冗余代码。

**Architecture:** 顶层 `common/` 包集中跨模块复用的领域逻辑（常量、关节筛选、骨龄公式、图像预处理），单一来源；`data_prepare/`、`train/`、`inference/` 三个模块只依赖 `common/`，通过文件与 argparse 解耦。检测模型走 ultralytics。

**Tech Stack:** Python 3.11、torch 2.6.0+cu126、torchvision 0.21.0、ultralytics 8.4.175、opencv-python、pillow、numpy、pytest。

**Spec:** `docs/superpowers/specs/2026-10-13-hand-bone-detect-restructure-design.md`

## Global Constraints

- Python 3.11，conda 环境 `hand_bone_detect`（`C:\project\envs\hand_bone_detect`）。
- 统一以模块方式从**仓库根**运行：`python -m <pkg>.<mod> ...`。仓库根的 `conftest.py` 保证 `pytest` 能 `import common`。
- **禁止硬编码绝对路径**；所有路径由 `argparse` 传入或基于 `__file__` 推导仓库根。
- 检测类别顺序必须与 `configs/bone_detect.yaml` 的 `names` 一致，且等于 `common.constants.DETECT_CLASSES`。
- 不改动骨龄算法本身：分数表、多项式系数、13 关节定义一律照搬现有 `common.py`。
- 每个任务结束前跑对应测试；测试全绿再进入下一任务。

---

### Task 1: `common/` 包骨架与常量

**Files:**
- Create: `conftest.py`
- Create: `tests/__init__.py`
- Create: `common/__init__.py`
- Create: `common/constants.py`
- Test: `tests/test_constants.py`
- Delete（稍后任务 13 统一处理）：`hand_bone_detect/common.py`（本任务只读它取常量）

**Interfaces:**
- Produces: `common.constants.DETECT_CLASSES: list[str]`、`common.constants.CATEGORY: list[str]`、`common.constants.ARTHROSIS: dict[str, list]`、`common.constants.SCORE: dict[str, dict[str, list[int]]]`

- [ ] **Step 1: 建 `conftest.py` 与包骨架**

创建空文件 `conftest.py`（仓库根，pytest 会自动把仓库根加入 `sys.path`）、`tests/__init__.py`、`common/__init__.py`（均为空）。

- [ ] **Step 2: 写失败测试 `tests/test_constants.py`**

```python
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
```

- [ ] **Step 3: 运行测试确认失败**

Run: `python -m pytest tests/test_constants.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'common'` 或 `constants` 不存在）

- [ ] **Step 4: 写 `common/constants.py`**

从 `hand_bone_detect/common.py` 原样搬入 `SCORE` 与 `arthrosis`（改名 `ARTHROSIS`）、`CATEGORY`，并补 `DETECT_CLASSES`（来自 `mydata.yaml` 的 names 顺序）：

```python
# 检测模型输出的 7 个骨骼类别，顺序必须与 configs/bone_detect.yaml 的 names 一致
DETECT_CLASSES = [
    "Radius",
    "Ulna",
    "MCPFirst",
    "MCP",
    "ProximalPhalanx",
    "MiddlePhalanx",
    "DistalPhalanx",
]

# 13 个目标关节的输出顺序（common.bone_filter.filter_joints 的输出顺序必须与此一致）
CATEGORY = [
    "DIPFifth", "DIPThird", "DIPFirst",
    "MIPFifth", "MIPThird",
    "PIPFifth", "PIPThird", "PIPFirst",
    "MCPFifth", "MCPThird", "MCPFirst",
    "Ulna", "Radius",
]

# 13 关节 -> (分类模型名, 分类类别数)
ARTHROSIS = {
    "MCPFirst": ["MCPFirst", 11],
    "MCPThird": ["MCP", 10],
    "MCPFifth": ["MCP", 10],
    "DIPFirst": ["DIPFirst", 11],
    "DIPThird": ["DIP", 11],
    "DIPFifth": ["DIP", 11],
    "PIPFirst": ["PIPFirst", 12],
    "PIPThird": ["PIP", 12],
    "PIPFifth": ["PIP", 12],
    "MIPThird": ["MIP", 12],
    "MIPFifth": ["MIP", 12],
    "Radius": ["Radius", 14],
    "Ulna": ["Ulna", 12],
}

SCORE = {
    "girl": {
        "Radius": [10, 15, 22, 25, 40, 59, 91, 125, 138, 178, 192, 199, 203, 210],
        "Ulna": [27, 31, 36, 50, 73, 95, 120, 157, 168, 176, 182, 189],
        "MCPFirst": [5, 7, 10, 16, 23, 28, 34, 41, 47, 53, 66],
        "MCPThird": [3, 5, 6, 9, 14, 21, 32, 40, 47, 51],
        "MCPFifth": [4, 5, 7, 10, 15, 22, 33, 43, 47, 51],
        "PIPFirst": [6, 7, 8, 11, 17, 26, 32, 38, 45, 53, 60, 67],
        "PIPThird": [3, 5, 7, 9, 15, 20, 25, 29, 35, 41, 46, 51],
        "PIPFifth": [4, 5, 7, 11, 18, 21, 25, 29, 34, 40, 45, 50],
        "MIPThird": [4, 5, 7, 10, 16, 21, 25, 29, 35, 43, 46, 51],
        "MIPFifth": [3, 5, 7, 12, 19, 23, 27, 32, 35, 39, 43, 49],
        "DIPFirst": [5, 6, 8, 10, 20, 31, 38, 44, 45, 52, 67],
        "DIPThird": [3, 5, 7, 10, 16, 24, 30, 33, 36, 39, 49],
        "DIPFifth": [5, 6, 7, 11, 18, 25, 29, 33, 35, 39, 49],
    },
    "boy": {
        "Radius": [8, 11, 15, 18, 31, 46, 76, 118, 135, 171, 188, 197, 201, 209],
        "Ulna": [25, 30, 35, 43, 61, 80, 116, 157, 168, 180, 187, 194],
        "MCPFirst": [4, 5, 8, 16, 22, 26, 34, 39, 45, 52, 66],
        "MCPThird": [3, 4, 5, 8, 13, 19, 30, 38, 44, 51],
        "MCPFifth": [3, 4, 6, 9, 14, 19, 31, 41, 46, 50],
        "PIPFirst": [4, 5, 7, 11, 17, 23, 29, 36, 44, 52, 59, 66],
        "PIPThird": [3, 4, 5, 8, 14, 19, 23, 28, 34, 40, 45, 50],
        "PIPFifth": [3, 4, 6, 10, 16, 19, 24, 28, 33, 40, 44, 50],
        "MIPThird": [3, 4, 5, 9, 14, 18, 23, 28, 35, 42, 45, 50],
        "MIPFifth": [3, 4, 6, 11, 17, 21, 26, 31, 36, 40, 43, 49],
        "DIPFirst": [4, 5, 6, 9, 19, 28, 36, 43, 46, 51, 67],
        "DIPThird": [3, 4, 5, 9, 15, 23, 29, 33, 37, 40, 49],
        "DIPFifth": [3, 4, 6, 11, 17, 23, 29, 32, 36, 40, 49],
    },
}
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_constants.py -v`
Expected: PASS（4 passed）

- [ ] **Step 6: 提交**

```bash
git add conftest.py tests/__init__.py tests/test_constants.py common/__init__.py common/constants.py
git commit -m "refactor: add common package with domain constants"
```

---

### Task 2: `common/bone_filter.py`（合并两份实现）

**Files:**
- Create: `common/bone_filter.py`
- Test: `tests/test_bone_filter.py`

**Interfaces:**
- Consumes: 无
- Produces: `common.bone_filter.filter_joints(boxes: torch.Tensor) -> torch.Tensor`，输入 `[N,6]` 列为 `[x1,y1,x2,y2,conf,cls]`，输出 `[13,6]`，顺序等于 `common.constants.CATEGORY`

- [ ] **Step 1: 写失败测试 `tests/test_bone_filter.py`**

21 个框 = DistalPhalanx×7、MiddlePhalanx×3、ProximalPhalanx×5、MCP×3、MCPFirst×1、Ulna×1、Radius×1；各类内 x1 递增，选取下标 `[0,2,4]`／`[0,2]`／`[0]`：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_bone_filter.py -v`
Expected: FAIL（`No module named 'common.bone_filter'`）

- [ ] **Step 3: 写 `common/bone_filter.py`**

采用与 `configs/bone_detect.yaml` 类别顺序自洽的那份（即原 `common.py` 版本），把内部 `filter` 改名 `_select` 避免遮蔽内置函数，并把计数异常改为抛错：

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_bone_filter.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add common/bone_filter.py tests/test_bone_filter.py
git commit -m "refactor: consolidate bone_filter into common package"
```

---

### Task 3: `common/bone_age.py`

**Files:**
- Create: `common/bone_age.py`
- Test: `tests/test_bone_age.py`

**Interfaces:**
- Consumes: 无
- Produces: `common.bone_age.calc_bone_age(score: float, sex: str) -> float`、`common.bone_age.export_report(results: dict, score: float, bone_age: float) -> str`

- [ ] **Step 1: 写失败测试 `tests/test_bone_age.py`**

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_bone_age.py -v`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 写 `common/bone_age.py`**

照搬原 `common.py` 的 `calcBoneAge`（改名 `calc_bone_age`）与 `export`（改名 `export_report`），系数一字不改：

```python
def calc_bone_age(score, sex):
    if sex == "boy":
        bone_age = (
            2.01790023656577
            - 0.0931820870747269 * score
            + score ** 2 * 0.00334709095418796
            + score ** 3 * (-3.32988302362153e-05)
            + score ** 4 * (1.75712910819776e-07)
            + score ** 5 * (-5.59998691223273e-10)
            + score ** 6 * (1.1296711294933e-12)
            + score ** 7 * (-1.45218037113138e-15)
            + score ** 8 * (1.15333377080353e-18)
            + score ** 9 * (-5.15887481551927e-22)
            + score ** 10 * (9.94098428102335e-26)
        )
        return round(bone_age, 2)
    if sex == "girl":
        bone_age = (
            5.81191794824917
            - 0.271546561737745 * score
            + score ** 2 * 0.00526301486340724
            + score ** 3 * (-4.37797717401925e-05)
            + score ** 4 * (2.0858722025667e-07)
            + score ** 5 * (-6.21879866563429e-10)
            + score ** 6 * (1.19909931745368e-12)
            + score ** 7 * (-1.49462900826936e-15)
            + score ** 8 * (1.162435538672e-18)
            + score ** 9 * (-5.12713017846218e-22)
            + score ** 10 * (9.78989966891478e-26)
        )
        return round(bone_age, 2)
    raise ValueError("sex 必须是 'boy' 或 'girl'，收到: {!r}".format(sex))


def export_report(results, score, bone_age):
    return """
    第一掌骨骺分级{}级，得{}分；第三掌骨骨骺分级{}级，得{}分；第五掌骨骨骺分级{}级，得{}分；
    第一近节指骨骨骺分级{}级，得{}分；第三近节指骨骨骺分级{}级，得{}分；第五近节指骨骨骺分级{}级，得{}分；
    第三中节指骨骨骺分级{}级，得{}分；第五中节指骨骨骺分级{}级，得{}分；
    第一远节指骨骨骺分级{}级，得{}分；第三远节指骨骨骺分级{}级，得{}分；第五远节指骨骨骺分级{}级，得{}分；
    尺骨分级{}级，得{}分；桡骨骨骺分级{}级，得{}分。

    RUS-CHN分级计分法，受检儿CHN总得分：{}分，骨龄约为{}岁。""".format(
        results['MCPFirst'][0], results['MCPFirst'][1],
        results['MCPThird'][0], results['MCPThird'][1],
        results['MCPFifth'][0], results['MCPFifth'][1],
        results['PIPFirst'][0], results['PIPFirst'][1],
        results['PIPThird'][0], results['PIPThird'][1],
        results['PIPFifth'][0], results['PIPFifth'][1],
        results['MIPThird'][0], results['MIPThird'][1],
        results['MIPFifth'][0], results['MIPFifth'][1],
        results['DIPFirst'][0], results['DIPFirst'][1],
        results['DIPThird'][0], results['DIPThird'][1],
        results['DIPFifth'][0], results['DIPFifth'][1],
        results['Ulna'][0], results['Ulna'][1],
        results['Radius'][0], results['Radius'][1],
        score, bone_age,
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_bone_age.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add common/bone_age.py tests/test_bone_age.py
git commit -m "refactor: move bone age formula into common package"
```

---

### Task 4: `common/image.py`（预处理统一）

**Files:**
- Create: `common/image.py`
- Test: `tests/test_image.py`

**Interfaces:**
- Consumes: 无
- Produces: `common.image.trans_square(image: PIL.Image.Image) -> PIL.Image.Image`、`common.image.clahe_detect(gray: np.ndarray) -> np.ndarray`（检测用 CLAHE）、`common.image.clahe_crop(gray: np.ndarray) -> np.ndarray`（分类裁剪用 CLAHE）

- [ ] **Step 1: 写失败测试 `tests/test_image.py`**

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_image.py -v`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 写 `common/image.py`**

`trans_square` 照搬现有实现；两种 CLAHE 参数分别对应"整张 X 光检测"与"关节裁剪分类"：

```python
import cv2
import numpy as np
from PIL import Image


def trans_square(image):
    """把 PIL 图片等比补黑边成正方形。"""
    img = image.convert("RGB")
    img = np.array(img, dtype=np.uint8)
    img_h, img_w, img_c = img.shape
    if img_h != img_w:
        long_side = max(img_w, img_h)
        short_side = min(img_w, img_h)
        loc = abs(img_w - img_h) // 2
        img = img.transpose((1, 0, 2)) if img_w < img_h else img
        background = np.zeros((long_side, long_side, img_c), dtype=np.uint8)
        background[loc:loc + short_side] = img[...]
        img = background.transpose((1, 0, 2)) if img_w < img_h else background
    return Image.fromarray(img, "RGB")


def clahe_detect(gray):
    """整张 X 光检测前的 CLAHE（参数与推理原有实现一致）。"""
    return cv2.createCLAHE(tileGridSize=(5, 5)).apply(gray)


def clahe_crop(gray):
    """关节裁剪图分类前的 CLAHE（参数与训练/推理原有实现一致）。"""
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3)).apply(gray)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_image.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add common/image.py tests/test_image.py
git commit -m "refactor: move image preprocessing into common package"
```

---

### Task 5: `data_prepare/voc_to_yolo.py`

**Files:**
- Create: `data_prepare/__init__.py`
- Create（由 `git mv` 改名后重写）: `data_prepare/voc_to_yolo.py`（源 `voc_to_yolo.py`）
- Test: `tests/test_voc_to_yolo.py`

**Interfaces:**
- Consumes: 无
- Produces: `data_prepare.voc_to_yolo.voc_to_yolo(xml_dir: str, out_dir: str, classes: list[str]) -> int`（返回转换数量）

- [ ] **Step 1: 写失败测试 `tests/test_voc_to_yolo.py`**

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_voc_to_yolo.py -v`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 建包并改名**

```bash
mkdir -p data_prepare
touch data_prepare/__init__.py
git mv voc_to_yolo.py data_prepare/voc_to_yolo.py
```

- [ ] **Step 4: 写 `data_prepare/voc_to_yolo.py`**

保留原有坐标换算公式（含 `-1` 与六位四舍五入），去掉硬编码路径，加 argparse：

```python
import argparse
import os
import xml.etree.ElementTree as ET

from tqdm import tqdm


def _convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (round(x * dw, 6), round(y * dh, 6), round(w * dw, 6), round(h * dh, 6))


def voc_to_yolo(xml_dir, out_dir, classes):
    os.makedirs(out_dir, exist_ok=True)
    names = [n for n in os.listdir(xml_dir) if n.endswith(".xml")]
    count = 0
    for name in tqdm(names):
        image_id = name[:-4]
        tree = ET.parse(os.path.join(xml_dir, name))
        root = tree.getroot()
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        lines = []
        for obj in root.iter("object"):
            if int(obj.find("difficult").text) == 1:
                continue
            cls = obj.find("name").text
            if cls not in classes:
                continue
            b = obj.find("bndbox")
            xmin = float(b.find("xmin").text)
            xmax = float(b.find("xmax").text)
            ymin = float(b.find("ymin").text)
            ymax = float(b.find("ymax").text)
            xmax = min(xmax, w)
            ymax = min(ymax, h)
            cx, cy, bw, bh = _convert((w, h), (xmin, xmax, ymin, ymax))
            lines.append("{} {} {} {} {}".format(classes.index(cls), cx, cy, bw, bh))
        with open(os.path.join(out_dir, image_id + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="VOC XML 标注转 YOLO txt")
    parser.add_argument("--xml-dir", required=True, help="VOC Annotations 目录")
    parser.add_argument("--out-dir", required=True, help="YOLO labels 输出目录")
    parser.add_argument("--classes", required=True, help="逗号分隔的类别名，顺序即类别 id")
    args = parser.parse_args()
    n = voc_to_yolo(args.xml_dir, args.out_dir, args.classes.split(","))
    print("转换完成，共 {} 个文件".format(n))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_voc_to_yolo.py -v`
Expected: PASS（2 passed）

- [ ] **Step 6: 提交**

```bash
git add data_prepare/ tests/test_voc_to_yolo.py
git commit -m "refactor: rewrite voc_to_yolo as path-parameterized CLI"
```

---

### Task 6: `data_prepare/split_dataset.py`

**Files:**
- Create（由 `git mv` 改名后重写）: `data_prepare/split_dataset.py`（源 `images_tag.py`，吸收 `transfer_files.py`）
- Delete: `transfer_files.py`（内容并入后删除）
- Test: `tests/test_split_dataset.py`

**Interfaces:**
- Consumes: 无
- Produces: `data_prepare.split_dataset.split_dataset(images_dir, labels_dir, out_root, trainval=0.9, train=0.7, seed=0) -> dict[str, int]`；输出 `out_root/images/{train,val,test}` 与 `out_root/labels/{train,val,test}`

- [ ] **Step 1: 写失败测试 `tests/test_split_dataset.py`**

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_split_dataset.py -v`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 建文件改名**

```bash
git mv images_tag.py data_prepare/split_dataset.py
```

- [ ] **Step 4: 写 `data_prepare/split_dataset.py`**

```python
import argparse
import os
import random
import shutil


def split_dataset(images_dir, labels_dir, out_root, trainval=0.9, train=0.7, seed=0):
    """按比例划分并拷贝成 images/{train,val,test} 与 labels/{train,val,test}。"""
    random.seed(seed)
    stems = [os.path.splitext(f)[0] for f in os.listdir(images_dir)]
    num = len(stems)
    tv = int(num * trainval)
    tr = int(tv * train)
    trainval_set = set(random.sample(stems, tv))
    train_set = set(random.sample(sorted(trainval_set), tr))

    buckets = {"train": [], "val": [], "test": []}
    for stem in stems:
        if stem not in trainval_set:
            buckets["test"].append(stem)
        elif stem in train_set:
            buckets["train"].append(stem)
        else:
            buckets["val"].append(stem)

    counts = {}
    for split, items in buckets.items():
        img_dir = os.path.join(out_root, "images", split)
        lbl_dir = os.path.join(out_root, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for stem in items:
            for ext in (".png", ".jpg", ".jpeg"):
                src = os.path.join(images_dir, stem + ext)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(img_dir, stem + ext))
                    break
            src_lbl = os.path.join(labels_dir, stem + ".txt")
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, os.path.join(lbl_dir, stem + ".txt"))
        counts[split] = len(items)
    return counts


def main():
    parser = argparse.ArgumentParser(description="划分 train/val/test 并整理目录")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--trainval", type=float, default=0.9)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    counts = split_dataset(
        args.images_dir, args.labels_dir, args.out_root,
        trainval=args.trainval, train=args.train, seed=args.seed,
    )
    print("划分完成:", counts)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_split_dataset.py -v`
Expected: PASS（1 passed）

- [ ] **Step 6: 删除已并入的旧脚本并提交**

```bash
git rm transfer_files.py
git add data_prepare/split_dataset.py tests/test_split_dataset.py
git commit -m "refactor: merge images_tag and transfer_files into split_dataset"
```

---

### Task 7: `data_prepare/prepare_classify.py`

**Files:**
- Create（由 `git mv` 改名后重写）: `data_prepare/prepare_classify.py`（源 `hand_test/utils/data_utils.py`）
- Test: `tests/test_prepare_classify.py`

**Interfaces:**
- Consumes: 无
- Produces: `data_prepare.prepare_classify.prepare_classify(data_root, train_ratio=0.9, seed=0) -> dict[str, dict[str, int]]`；在**每个关节子目录**下生成 `train.txt`/`val.txt`，每行 `图片绝对路径 标签`（标签 = 等级目录名 - 1）

- [ ] **Step 1: 写失败测试 `tests/test_prepare_classify.py`**

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_prepare_classify.py -v`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 建文件改名**

```bash
git mv hand_test/utils/data_utils.py data_prepare/prepare_classify.py
```

- [ ] **Step 4: 写 `data_prepare/prepare_classify.py`**

保留"去雾 + 旋转增样"两个可选函数，主流程只做切分与生成 txt；去掉脚本顶层的自动执行副作用：

```python
import argparse
import os
import random

import cv2
from PIL import Image


def opt_img(img_path):
    """对单张图做 CLAHE 去雾（原地覆盖）。"""
    img = cv2.imread(img_path, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    cv2.imwrite(img_path, clahe.apply(img))


def img_rotate(img_path, flag=5):
    """旋转增样，生成 flag 张副本。"""
    img = Image.open(img_path)
    for i in range(flag):
        dst = img.rotate(random.randint(-45, 45))
        base, _ = os.path.splitext(img_path)
        dst.save("{}_aug{}.png".format(base, i))


def _write_list(path, lines):
    if os.path.exists(path):
        os.remove(path)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def prepare_classify(data_root, train_ratio=0.9, seed=0):
    """data_root 下每个关节目录包含 1..N 个等级子目录，切分并生成 train/val.txt。"""
    random.seed(seed)
    stats = {}
    for joint in os.listdir(data_root):
        joint_dir = os.path.join(data_root, joint)
        if not os.path.isdir(joint_dir):
            continue
        train_list, val_list = [], []
        for grade in os.listdir(joint_dir):
            grade_dir = os.path.join(joint_dir, grade)
            if not os.path.isdir(grade_dir):
                continue
            images = os.listdir(grade_dir)
            random.shuffle(images)
            cut = int(len(images) * train_ratio)
            for idx, img in enumerate(images):
                line = "{} {}\n".format(os.path.join(grade_dir, img), int(grade) - 1)
                (train_list if idx < cut else val_list).append(line)
        random.shuffle(train_list)
        random.shuffle(val_list)
        _write_list(os.path.join(joint_dir, "train.txt"), train_list)
        _write_list(os.path.join(joint_dir, "val.txt"), val_list)
        stats[joint] = {"train": len(train_list), "val": len(val_list)}
        print("{}: train={} val={}".format(joint, len(train_list), len(val_list)))
    return stats


def main():
    parser = argparse.ArgumentParser(description="分类数据划分，生成 train/val.txt")
    parser.add_argument("--data-root", required=True, help="关节数据根目录")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--augment", action="store_true", help="额外做去雾 + 旋转增样")
    args = parser.parse_args()
    if args.augment:
        for joint in os.listdir(args.data_root):
            joint_dir = os.path.join(args.data_root, joint)
            for grade in os.listdir(joint_dir):
                grade_dir = os.path.join(joint_dir, grade)
                if os.path.isdir(grade_dir):
                    for img in os.listdir(grade_dir):
                        p = os.path.join(grade_dir, img)
                        opt_img(p)
                        img_rotate(p)
    prepare_classify(args.data_root, train_ratio=args.train_ratio, seed=args.seed)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_prepare_classify.py -v`
Expected: PASS（1 passed）

- [ ] **Step 6: 提交**

```bash
git add data_prepare/prepare_classify.py tests/test_prepare_classify.py
git commit -m "refactor: move classification data prep into data_prepare"
```

---

### Task 8: `train/classify/`（分类训练）

**Files:**
- Create: `train/__init__.py`、`train/classify/__init__.py`
- Create（由 `git mv` 改名后重写）: `train/classify/dataset.py`（源 `hand_test/utils/data_set.py`）
- Create（由 `git mv` 改名后重写）: `train/classify/train_classify.py`（源 `hand_test/trainer.py`）
- Test: `tests/test_classify_dataset.py`

**Interfaces:**
- Consumes: `common.image.trans_square`、`common.constants.ARTHROSIS`
- Produces: `train.classify.dataset.My_data(file_dir: str, mode: str)`；`train.classify.train_classify.create_model(num_classes: int) -> nn.Module`、`train_classify(category: str, data_root: str) -> None`

- [ ] **Step 1: 写失败测试 `tests/test_classify_dataset.py`**

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_classify_dataset.py -v`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 建包并改名**

```bash
mkdir -p train/classify
touch train/__init__.py train/classify/__init__.py
git mv hand_test/utils/data_set.py train/classify/dataset.py
git mv hand_test/trainer.py train/classify/train_classify.py
```

- [ ] **Step 4: 写 `train/classify/dataset.py`**

```python
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, InterpolationMode, Resize, ToTensor

from common.image import trans_square

data_transforms = Compose([
    Resize(size=(224, 224), interpolation=InterpolationMode.NEAREST),
    ToTensor(),
])


class My_data(Dataset):
    def __init__(self, file_dir, mode):
        self.data_list = []
        file_path = os.path.join(file_dir, "train.txt" if mode == "train" else "val.txt")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                items = line.strip().split()
                if not items:
                    continue
                self.data_list.append([items[0], int(items[1])])

    def __getitem__(self, idx):
        image_path, label = self.data_list[idx]
        im = Image.open(image_path)
        im = trans_square(im)
        im = im.convert("L")
        im = data_transforms(im)
        return im, torch.tensor(label)

    def __len__(self):
        return len(self.data_list)
```

- [ ] **Step 5: 写 `train/classify/train_classify.py`**

逻辑照搬原 `trainer.py`，路径改参数、类别表改从 `common.constants.ARTHROSIS` 取、按 best val acc 保存到 `models/classify/`：

```python
import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from common.constants import ARTHROSIS
from train.classify.dataset import My_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAMES = list(dict.fromkeys(v[0] for v in ARTHROSIS.values()))
NUM_CLASSES = {v[0]: v[1] for v in ARTHROSIS.values()}


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_model(num_classes):
    net = models.resnet18()
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net.fc = nn.Linear(512, num_classes)
    return net.to(DEVICE)


def train_classify(category, data_root, out_dir, epochs=80, batch_size=60):
    num_classes = NUM_CLASSES[category]
    train_set = My_data(os.path.join(data_root, category), "train")
    val_set = My_data(os.path.join(data_root, category), "val")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    model = create_model(num_classes)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "{}_best.pth".format(category))

    loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters())
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for img, label in train_loader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            loss = loss_func(model(img), label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        model.eval()
        acc_sum, n = 0.0, 0
        with torch.no_grad():
            for img, label in val_loader:
                img, label = img.to(DEVICE), label.to(DEVICE)
                out = model(img)
                acc_sum += torch.mean(torch.eq(out.argmax(dim=1), label).float()).item()
                n += 1
        val_acc = acc_sum / max(n, 1)
        print("epoch={} train_loss={:.4f} val_acc={:.4f}".format(epoch, train_loss / max(len(train_loader), 1), val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt)
            print("保存模型:", ckpt)


def main():
    parser = argparse.ArgumentParser(description="训练 9 个关节分级分类模型")
    parser.add_argument("--data-root", required=True, help="分类数据根目录（各关节子目录）")
    parser.add_argument("--out-dir", default=os.path.join(_repo_root(), "models", "classify"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--category", help="只训练指定关节（默认全部）")
    args = parser.parse_args()
    categories = [args.category] if args.category else MODEL_NAMES
    for cat in categories:
        print("开始训练:", cat)
        train_classify(cat, args.data_root, args.out_dir, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m pytest tests/test_classify_dataset.py -v`
Expected: PASS（2 passed）

- [ ] **Step 7: 提交**

```bash
git add train/ tests/test_classify_dataset.py
git commit -m "refactor: move classification training into train/classify"
```

---

### Task 9: `train/detect/` 与数据集配置

**Files:**
- Create: `train/detect/__init__.py`
- Create: `train/detect/train_detect.py`
- Create: `configs/bone_detect.yaml`
- Test: `tests/test_config.py`
- Delete: `mydata.yaml`（改名迁移后）

**Interfaces:**
- Consumes: `common.constants.DETECT_CLASSES`
- Produces: `train.detect.train_detect.train_detect(data_yaml, weights, epochs, batch, imgsz, project) -> str`（返回 best 权重路径）

- [ ] **Step 1: 写失败测试 `tests/test_config.py`**

```python
import yaml

from common.constants import DETECT_CLASSES


def test_config_names_match_detect_classes():
    with open("configs/bone_detect.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    assert [names[i] for i in range(len(names))] == DETECT_CLASSES
    for key in ("path", "train", "val"):
        assert key in cfg
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL（找不到文件）

- [ ] **Step 3: 写 `configs/bone_detect.yaml`**

```yaml
path: ../datasets/VOCdevkit
train: images/train
val: images/val
test: images/test

names:
  0: Radius
  1: Ulna
  2: MCPFirst
  3: MCP
  4: ProximalPhalanx
  5: MiddlePhalanx
  6: DistalPhalanx
```

- [ ] **Step 4: 写 `train/detect/train_detect.py`**

```python
import argparse
import os

from ultralytics import YOLO


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_detect(data_yaml, weights, epochs, batch, imgsz, project):
    model = YOLO(weights)
    model.train(data=data_yaml, epochs=epochs, batch=batch, imgsz=imgsz, project=project)
    return str(model.trainer.best)


def main():
    root = _repo_root()
    parser = argparse.ArgumentParser(description="用 ultralytics 训练骨骼检测模型")
    parser.add_argument("--data", default=os.path.join(root, "configs", "bone_detect.yaml"))
    parser.add_argument("--weights", default=os.path.join(root, "models", "yolov5su.pt"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default=os.path.join(root, "runs", "detect"))
    parser.add_argument("--name", default="bone_detect")
    args = parser.parse_args()
    best = train_detect(args.data, args.weights, args.epochs, args.batch, args.imgsz, args.project)
    print("训练完成，best:", best)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS（1 passed）

- [ ] **Step 6: 提交**

```bash
git rm mydata.yaml
git add train/detect/ configs/bone_detect.yaml tests/test_config.py
git commit -m "feat: add ultralytics detection training and dataset config"
```

---

### Task 10: `inference/detector.py`

**Files:**
- Create: `inference/__init__.py`
- Create: `inference/detector.py`
- Test: `tests/test_detector.py`

**Interfaces:**
- Consumes: `common.image.clahe_detect`
- Produces: `inference.detector.load_detector(weights=None) -> YOLO`、`inference.detector.detect(model, img_path, conf=0.7) -> tuple[np.ndarray, torch.Tensor]`（返回 `(BGR 原图, boxes[N,6])`）

- [ ] **Step 1: 写失败测试 `tests/test_detector.py`**

用 stub 验证 `detect` 的预处理与返回格式，不依赖真实权重：

```python
import numpy as np
import torch

from inference import detector


class _StubPredictor:
    """鸭子类型替身：记录输入，返回固定的 ultralytics 风格结果。"""

    def __init__(self):
        self.seen_conf = None
        self.seen_input_shape = None

    def predict(self, image, conf=0.25, verbose=False):
        self.seen_conf = conf
        self.seen_input_shape = image.shape

        class _Boxes:
            data = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.9, 0.0]])

        class _Result:
            boxes = _Boxes()

        return [_Result()]


def test_detect_returns_image_and_boxes(tmp_path):
    import cv2

    img_path = tmp_path / "x.png"
    cv2.imwrite(str(img_path), np.zeros((30, 40, 3), dtype=np.uint8))

    stub = _StubPredictor()
    img, boxes = detector.detect(stub, str(img_path), conf=0.5)

    assert img.shape == (30, 40, 3)
    assert boxes.shape == (1, 6)
    assert stub.seen_conf == 0.5
    assert stub.seen_input_shape == (30, 40)   # 送入的是灰度图
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_detector.py -v`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 写 `inference/detector.py`**

```python
import os

import cv2
from ultralytics import YOLO

from common.image import clahe_detect


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DEFAULT_WEIGHTS = os.path.join(_repo_root(), "models", "yolov5su.pt")


def load_detector(weights=None):
    weights = weights or DEFAULT_WEIGHTS
    if not os.path.exists(weights):
        raise FileNotFoundError("检测权重不存在: {}，请先训练或下载".format(weights))
    return YOLO(weights)


def detect(model, img_path, conf=0.7):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = model.predict(clahe_detect(gray), conf=conf, verbose=False)[0]
    return img, result.boxes.data
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_detector.py -v`
Expected: PASS（1 passed）

- [ ] **Step 5: 提交**

```bash
git add inference/__init__.py inference/detector.py tests/test_detector.py
git commit -m "feat: add ultralytics detector wrapper"
```

---

### Task 11: `inference/classifier.py` 与 `inference/pipeline.py`

**Files:**
- Create: `inference/classifier.py`
- Create: `inference/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `common.bone_filter.filter_joints`、`common.bone_age.calc_bone_age`、`common.bone_age.export_report`、`common.constants.{CATEGORY,ARTHROSIS,SCORE}`、`common.image.{trans_square,clahe_crop}`
- Produces:
  - `inference.classifier.load_classifiers(base_dir=None) -> dict[str, nn.Module]`
  - `inference.classifier.classify_crop(model, pil_img) -> int`（0 起的等级）
  - `inference.pipeline.predict_bone_age(img_path, sex, detector, classifiers, conf=0.7) -> str`
  - 约定：`detector` 是"有 `.detect(img_path, conf)` 方法、返回 `(img, boxes)`"的对象（`inference.detector` 模块本身或替身均可）

- [ ] **Step 1: 写失败测试 `tests/test_pipeline.py`**

用替身检测器验证编排与异常路径，不依赖真实权重：

```python
import numpy as np
import pytest
import torch

from inference import pipeline


class _FakeDetector:
    def __init__(self, boxes):
        self._boxes = boxes

    def detect(self, img_path, conf=0.7):
        return np.zeros((100, 100, 3), dtype=np.uint8), self._boxes


def test_predict_raises_on_wrong_box_count(tmp_path):
    fake = _FakeDetector(torch.zeros((5, 6)))
    with pytest.raises(ValueError) as e:
        pipeline.predict_bone_age("x.png", "boy", fake, {})
    assert "预期 21" in str(e.value)


def test_predict_full_path_with_stub_classifier(tmp_path, monkeypatch):
    rows = []
    for x in (11, 12, 13, 14, 15, 16, 17):
        rows.append([x, 0, x + 5, 10, 0.9, 6])
    for x in (21, 22, 23):
        rows.append([x, 0, x + 5, 10, 0.9, 5])
    for x in (31, 32, 33, 34, 35):
        rows.append([x, 0, x + 5, 10, 0.9, 4])
    for x in (41, 42, 43):
        rows.append([x, 0, x + 5, 10, 0.9, 3])
    rows += [[51, 0, 56, 10, 0.9, 2], [61, 0, 66, 10, 0.9, 1], [71, 0, 76, 10, 0.9, 0]]
    fake = _FakeDetector(torch.tensor(rows, dtype=torch.float32))

    # 每个分类模型恒返回等级 0
    monkeypatch.setattr(pipeline, "classify_crop", lambda model, crop: 0)
    report = pipeline.predict_bone_age("x.png", "boy", fake, {name: object() for name in ("MCP", "MCPFirst", "DIP", "DIPFirst", "PIP", "PIPFirst", "MIP", "Radius", "Ulna")})

    assert "骨龄约为" in report
    assert "总得分" in report
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 写 `inference/classifier.py`**

```python
import os

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models
from torchvision.transforms import Compose, InterpolationMode, Resize, ToTensor

from common.constants import ARTHROSIS
from common.image import clahe_crop, trans_square

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAMES = list(dict.fromkeys(v[0] for v in ARTHROSIS.values()))
NUM_CLASSES = {v[0]: v[1] for v in ARTHROSIS.values()}

data_transforms = Compose([
    Resize(size=(224, 224), interpolation=InterpolationMode.NEAREST),
    ToTensor(),
])


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_classifiers(base_dir=None):
    base_dir = base_dir or os.path.join(_repo_root(), "models", "classify")
    cls_models = {}
    for name in MODEL_NAMES:
        path = os.path.join(base_dir, "{}_best.pth".format(name))
        if not os.path.exists(path):
            raise FileNotFoundError("分类权重不存在: {}，请先训练".format(path))
        net = models.resnet18()
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        net.fc = nn.Linear(512, NUM_CLASSES[name])
        net.load_state_dict(torch.load(path, map_location=DEVICE))
        net.eval()
        cls_models[name] = net
    return cls_models


def classify_crop(model, pil_img):
    """对单个关节裁剪图分级，返回 0 起的等级。"""
    im = trans_square(pil_img).convert("L")
    im = clahe_crop(np.array(im))
    im = data_transforms(Image.fromarray(im))
    im = torch.unsqueeze(im, dim=0)
    with torch.no_grad():
        out = model(im)
    return int(out.argmax(dim=1))
```

- [ ] **Step 4: 写 `inference/pipeline.py`**

```python
from PIL import Image

from common.bone_age import calc_bone_age, export_report
from common.bone_filter import filter_joints
from common.constants import ARTHROSIS, CATEGORY, SCORE
from inference.classifier import classify_crop


def predict_bone_age(img_path, sex, detector, classifiers, conf=0.7):
    """端到端：检测 -> 筛 13 关节 -> 裁剪分级 -> 查分表 -> 算骨龄。"""
    img, boxes = detector.detect(img_path, conf=conf)
    joints = filter_joints(boxes).cpu().numpy()

    score_all = 0
    bone_results = {}
    for name, box in zip(CATEGORY, joints):
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        crop = img[y1:y2, x1:x2]
        crop = Image.fromarray(crop[:, :, ::-1])  # BGR -> RGB
        model = classifiers[ARTHROSIS[name][0]]
        grade = classify_crop(model, crop)
        score = SCORE[sex][name][grade]
        bone_results[name] = [grade + 1, int(score)]
        score_all += score

    bone_age = calc_bone_age(score_all, sex)
    return export_report(bone_results, score_all, bone_age)
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: PASS（2 passed）

- [ ] **Step 6: 提交**

```bash
git add inference/classifier.py inference/pipeline.py tests/test_pipeline.py
git commit -m "feat: add classifier wrapper and end-to-end bone age pipeline"
```

---

### Task 12: `inference/predict.py`（CLI 入口）

**Files:**
- Create（由 `git mv` 改名后重写）: `inference/predict.py`（源 `hand_bone_detect/detect_bone.py`）
- Delete: `hand_bone_detect/detect_utils.py`（能力已拆分到 detector/classifier/pipeline）
- Test: `tests/test_predict_cli.py`

**Interfaces:**
- Consumes: `inference.detector.load_detector`、`inference.classifier.load_classifiers`、`inference.pipeline.predict_bone_age`
- Produces: 命令行 `python -m inference.predict --img <path> --sex boy|girl [--conf 0.7] [--weights ...] [--classify-dir ...]`

- [ ] **Step 1: 写失败测试 `tests/test_predict_cli.py`**

验证"框数异常时给出友好报错并退出码非 0"，用替身避免真实权重：

```python
import numpy as np
import torch

from inference import predict


def test_run_reports_box_count_error(monkeypatch, capsys):
    class _FakeDetector:
        def detect(self, img_path, conf=0.7):
            return np.zeros((10, 10, 3), dtype=np.uint8), torch.zeros((7, 6))

    monkeypatch.setattr(predict, "load_detector", lambda weights=None: _FakeDetector())
    monkeypatch.setattr(predict, "load_classifiers", lambda base_dir=None: {})

    code = predict._run("x.png", "boy", 0.7, None, None)

    assert code == 1
    assert "预期 21" in capsys.readouterr().err
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_predict_cli.py -v`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 建文件改名**

```bash
git mv hand_bone_detect/detect_bone.py inference/predict.py
git rm hand_bone_detect/detect_utils.py
```

- [ ] **Step 4: 写 `inference/predict.py`**

```python
import argparse

from inference.classifier import load_classifiers
from inference.detector import load_detector
from inference.pipeline import predict_bone_age


def _run(img_path, sex, conf, weights, classify_dir):
    try:
        detector = load_detector(weights)
        classifiers = load_classifiers(classify_dir)
        report = predict_bone_age(img_path, sex, detector, classifiers, conf=conf)
    except ValueError as e:
        print("错误: {}".format(e), file=__import__("sys").stderr)
        return 1
    print(report)
    return 0


def main():
    parser = argparse.ArgumentParser(description="手部 X 光片骨龄预测")
    parser.add_argument("--img", required=True, help="输入 X 光片路径")
    parser.add_argument("--sex", required=True, choices=["boy", "girl"])
    parser.add_argument("--conf", type=float, default=0.7, help="检测置信度阈值")
    parser.add_argument("--weights", default=None, help="检测权重路径")
    parser.add_argument("--classify-dir", default=None, help="分类权重目录")
    args = parser.parse_args()
    raise SystemExit(_run(args.img, args.sex, args.conf, args.weights, args.classify_dir))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_predict_cli.py -v`
Expected: PASS（1 passed）

- [ ] **Step 6: 提交**

```bash
git add inference/predict.py tests/test_predict_cli.py
git commit -m "feat: add bone age prediction CLI"
```

---

### Task 13: 删除冗余、迁移权重、补文档

**Files:**
- Delete: `hand_bone_detect/`（整个目录：`flask_test_img.py`、`main.py`、`bone_detect_ui.ui`、`templates/`、`test.py`、`bone_filter_utils.py`、`img/`、`test_data/`、`cutpictures/`、`detect_result/`、`__pycache__/`、`common.py`）
- Delete: `res.mhtml`、`hand_test/`（`params/` 迁移后整个目录清理，含 `utils/tools.py`）
- Move: `hand_test/params/*.pth` → `models/classify/`（并解除 git 跟踪）
- Create: `requirements.txt`
- Modify: `README.md`、`CLAUDE.md`

**Interfaces:**
- Produces: 干净的最终目录结构与 `models/classify/*.pth`（9 个）

- [ ] **Step 1: 迁移分类权重并解除跟踪**

```bash
mkdir -p models/classify
git mv hand_test/params/DIPFirst_best.pth models/classify/DIPFirst_best.pth
git mv hand_test/params/DIP_best.pth models/classify/DIP_best.pth
git mv hand_test/params/MCPFirst_best.pth models/classify/MCPFirst_best.pth
git mv hand_test/params/MCP_best.pth models/classify/MCP_best.pth
git mv hand_test/params/MIP_best.pth models/classify/MIP_best.pth
git mv hand_test/params/PIPFirst_best.pth models/classify/PIPFirst_best.pth
git mv hand_test/params/PIP_best.pth models/classify/PIP_best.pth
git mv hand_test/params/Radius_best.pth models/classify/Radius_best.pth
git mv hand_test/params/Ulna_best.pth models/classify/Ulna_best.pth
git rm -r --cached --ignore-unmatch models/classify
```

> 若 `git mv` 因目标在 `.gitignore` 中报错，改为：`mv hand_test/params/*.pth models/classify/` 后再 `git rm -r --cached hand_test/params`。完成后 `git status` 应显示 9 个 `.pth` 从跟踪中移除，且 `models/` 不出现在 untracked 中。

- [ ] **Step 2: 删除冗余文件与目录**

```bash
git rm -r --ignore-unmatch hand_bone_detect
git rm -r --ignore-unmatch hand_test
git rm --ignore-unmatch res.mhtml
```

- [ ] **Step 3: 写 `requirements.txt`**

```
torch==2.6.0
torchvision==0.21.0
ultralytics==8.4.175
opencv-python
pillow
numpy
pyyaml
tqdm
pytest
```

- [ ] **Step 4: 重写 `README.md`**

按新结构写：项目简介（检测→分级→骨龄）、环境安装（Python 3.11 + torch cu126 + ultralytics）、三大模块的用法示例：

```markdown
# 骨龄检测

手部 X 光片 → 检测 21 个骨骼框 → 筛选 13 个目标关节 → 分类分级 → RUS-CHN 计分 → 骨龄。

## 环境

- Python 3.11
- `pip install -r requirements.txt`
- GPU 版 torch：`pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126`

## 模块

### ① 数据处理
python -m data_prepare.voc_to_yolo --xml-dir <VOC Annotations> --out-dir <labels> --classes Radius,Ulna,MCPFirst,MCP,ProximalPhalanx,MiddlePhalanx,DistalPhalanx
python -m data_prepare.split_dataset --images-dir <JPEGImages> --labels-dir <labels> --out-root <VOCdevkit>
python -m data_prepare.prepare_classify --data-root <关节数据根目录>

### ② 训练
python -m train.detect.train_detect --data configs/bone_detect.yaml --weights models/yolov5su.pt
python -m train.classify.train_classify --data-root <分类数据根目录>

### ③ 推理
python -m inference.predict --img <X光片> --sex boy

## 测试
python -m pytest tests/ -v
```

- [ ] **Step 5: 更新 `CLAUDE.md`**

把架构说明改为新的三模块结构，删除对 `hand_bone_detect/`、`hand_test/`、`bone_filter_utils.py`、硬编码路径的描述，补上"脚本以 `python -m` 从仓库根运行""检测类别顺序须与 `configs/bone_detect.yaml` 一致"。

- [ ] **Step 6: 全量测试 + 冒烟**

Run: `python -m pytest tests/ -v`
Expected: 全部 PASS

Run: `python -m inference.predict --img <任意手部X光或合成图> --sex boy`
Expected: 因当前检测权重是 COCO 版 `yolov5su.pt`，输出 `错误: 检测到 N 个框，预期 21，请检查检测模型/置信度阈值`（友好报错、退出码 1），而非 traceback。

- [ ] **Step 7: 提交**

```bash
git add -A
git commit -m "chore: remove legacy code, relocate weights, refresh docs"
```

---

## Self-Review

**Spec 覆盖检查：**
- 顶层三模块结构 → Task 5-12 建起 `data_prepare/`、`train/`、`inference/`，Task 13 清理旧目录 ✔
- 共享核心包 → Task 1-4 ✔
- 数据处理（检测+分类）→ Task 5-7 ✔
- 训练两部分 → Task 8（分类）、Task 9（检测）✔
- 推理全链路 → Task 10-12 ✔
- 删除清单 → Task 13 ✔
- 迁移清单（git mv）→ Task 5-9、13 ✔
- 无硬编码绝对路径 → 各任务均用 argparse + `__file__` ✔
- 错误处理（框数≠21、权重缺失）→ Task 2、10、11、12 ✔
- 最小单测 → Task 1-12 各带测试 ✔
- 验收标准 → Task 13 Step 6 ✔

**类型一致性检查：** `filter_joints`（Task 2）→ 在 Task 11 pipeline 以 `filter_joints(boxes).cpu().numpy()` 使用，签名一致；`classify_crop(model, pil_img)`（Task 11 classifier）→ pipeline 以 `classify_crop(model, crop)` 调用一致；`detect(model, img_path, conf)`（Task 10）与替身/`_FakeDetector` 的 `.detect(img_path, conf)` 一致；`ARTHROSIS`/`SCORE`/`CATEGORY` 名称在 Task 1 定义、Task 8/11 引用一致；`NUM_CLASSES`（model 名→类别数）在 Task 8 与 Task 11 各自按同一规则从 `ARTHROSIS` 派生，取值一致 ✔
