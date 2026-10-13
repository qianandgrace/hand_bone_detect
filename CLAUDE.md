# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于手部 X 光片预测骨龄（RSNA 2017 Bone Age 数据集）。整体流程模拟医生操作：

检测关节 → 按位置筛选目标关节 → 切割关节 → 对每个关节分类分级 → 查表得分为 → 代入公式计算骨龄。

这是一个**两阶段模型流水线**：

1. **YOLOv5 检测**（需单独 clone 的外部仓库 `yolov5/`，已 gitignore）：检测 7 类骨骼目标，一张手片应输出 21 个框。
2. **ResNet18 分类**：对切割出的关节做成熟度分级（每个关节一个模型）。

## 常用命令

数据准备（根目录，脚本内含硬编码 Windows 绝对路径，换机器需改）：

```bash
python images_tag.py       # 划分 train/val/test 列表
python voc_to_yolo.py      # VOC XML -> YOLO txt 标签
python transfer_files.py   # 按列表把图片/标签拷入 images/{train,val,test}
```

YOLOv5 检测训练（在 `yolov5/` 仓库内执行）：

```bash
mv mydata.yaml yolov5/data/
python train.py --batch 16 --epochs 300 --data mydata.yaml --weights yolov5s.pt
```

分类模型训练（一次性训练全部 9 个分类器，逐个保存 best）：

```bash
python hand_test/trainer.py
```

推理 / 部署（**必须在 `hand_bone_detect/` 目录内运行**，权重路径为相对路径 `../hand_test/params/`）：

```bash
cd hand_bone_detect
python detect_bone.py       # 单张图片检测（改 detect_bone.py 里的 img_path）
python flask_test_img.py    # Flask 服务，端口 5000，上传图片+性别返回骨龄
python main.py              # PyQt5 桌面 GUI（bone_detect_ui.ui）
```

## 架构与关键文件

### 领域常量集中地：`hand_bone_detect/common.py`

改类别、分数表、骨龄公式、关节映射时**只改这里**：

- `bone_filter`：把 YOLO 的 21 个框筛选/排序成 **13 个目标关节**（按 x 坐标排序取指定指骨）。
- `arthrosis`：13 个关节 → 9 个分类模型（模型名 + 分类数）的映射。
- `CATEGORY`：13 个关节的输出顺序，**必须与 `bone_filter` 的输出顺序一致**。
- `SCORE`：boy/girl × 关节 × 等级的分数表（RUS-CHN 计分法）。
- `calcBoneAge`：由总得分拟合骨龄的多项式（男女不同）。
- `export`：生成中文结论文本。
- `trans_square`：等比补黑边成正方形（分类模型输入前处理）。

### 部署层 `hand_bone_detect/`

- `detect_utils.py`：`load_model()` 加载 1 个 YOLOv5 + 9 个 ResNet18；`detect_img()` 是推理主流程（Flask 与 PyQt 都调用它）。
- `detect_bone.py`：脚本版入口，逻辑与 `detect_utils` 重复。
- `flask_test_img.py` / `main.py`：Flask 与 PyQt5 两种前端。
- `bone_filter_utils.py`：另一份 `bone_filter` 实现（见下方注意事项）。

### 训练层 `hand_test/`

- `trainer.py`：ResNet18（输入改单通道灰度、输出改为该关节级数）训练循环，学习率优化器 AdamW，按 val acc 保存 `params/{category}_best.pth`。
- `utils/data_set.py`：`My_data` Dataset，读 `{split}.txt`（每行 `图片路径 标签`）。
- `utils/data_utils.py`：CLIP 去雾预处理 + 旋转增样 + 划分 train/val 并生成 txt（**脚本本身会直接执行**，路径写死 `D:/dataset/...`）。
- `utils/tools.py`：`trans_square`（与 common.py 中的实现重复）。

### 数据流约定

YOLO 类别顺序由 `mydata.yaml` 定义：
`0=Radius, 1=Ulna, 2=MCPFirst, 3=MCP, 4=ProximalPhalanx, 5=MiddlePhalanx, 6=DistalPhalanx`。
`bone_filter` 依据该类别号 + 框的 x 坐标排序挑选关节，因此**类别号顺序与排序位置强耦合**，改动其一需同步改动另一处。

## 注意事项

- **`bone_filter` 有两个版本且不一致**：`common.py` 的版本与 `mydata.yaml` 类别顺序自洽；`bone_filter_utils.py` 的版本类别映射与输出顺序都不同，而 `detect_utils.py` / `detect_bone.py` 实际 import 的是 `bone_filter_utils` 里的这一份。改动筛选逻辑时务必确认用的是哪一份。
- **大量硬编码绝对路径**：`detect_utils.py`/`detect_bone.py` 里 YOLOv5 仓库与 `best.pt` 路径写死为 `C:\Users\qian gao\...`；`trainer.py`/`data_utils.py`/`data_set.py` 用 `D:/dataset/...`。换机器/目录时这些都要改。
- **外部依赖未随仓库提供**：`yolov5/` 与 `datasets/` 都在 `.gitignore` 中，需要按 README 自行 clone / 准备数据。YOLOv5 权重放在 `yolov5/runs/train/exp6/weights/best.pt`。
- **环境**：torch 2.5.1 + cu124（GPU 版），安装见 README；Win 上默认装的是 CPU 版，需按官网命令换 GPU 版。
- 分类模型推理时对裁剪图做了 CLAHE 增强（`detect_utils.detect_img` 里 clipLimit=2.0, tileGridSize=(3,3)），训练时（`data_utils.opt_img`）用的是相同参数，两者需保持一致。
