# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于手部 X 光片预测骨龄（RSNA 2017 Bone Age 数据集）。整体流程模拟医生操作：

检测关节 → 按位置筛选目标关节 → 切割关节 → 对每个关节分类分级 → 查表得分 → 代入公式计算骨龄。

这是一个**两阶段模型流水线**：

1. **YOLO 检测**（`ultralytics`，权重 `models/yolov5su.pt`）：检测 7 类骨骼目标，一张手片应输出 21 个框。
2. **ResNet18 分类**（输入单通道灰度，权重 `models/classify/{category}_best.pth`）：对切割出的关节做成熟度分级（每个关节一个模型）。

## 目录结构

```
common/        共享核心包（常量、骨龄计算、关节筛选、图像处理）
data_prepare/  数据处理（VOC→YOLO、数据集划分、分类数据准备）
train/         训练（train.detect 检测训练、train.classify 分类训练）
inference/     推理（detector、classifier、pipeline、predict CLI）
configs/       配置（bone_detect.yaml 检测类别定义）
tests/         单元测试
models/        权重（已被 .gitignore 忽略）
```

## 常用命令

所有脚本均以 `python -m <模块>` 从**仓库根目录**运行（不要 cd 进子目录），路径参数用命令行传入，无硬编码绝对路径。

数据处理：

```bash
python -m data_prepare.voc_to_yolo --xml-dir <VOC Annotations> --out-dir <labels> --classes Radius,Ulna,MCPFirst,MCP,ProximalPhalanx,MiddlePhalanx,DistalPhalanx
python -m data_prepare.split_dataset --images-dir <JPEGImages> --labels-dir <labels> --out-root <VOCdevkit>
python -m data_prepare.prepare_classify --data-root <关节数据根目录>
```

检测训练：

```bash
python -m train.detect.train_detect --data configs/bone_detect.yaml --weights models/yolov5su.pt
```

分类训练（一次性训练全部 9 个分类器，逐个保存 best）：

```bash
python -m train.classify.train_classify --data-root <分类数据根目录>
```

推理：

```bash
python -m inference.predict --img <X光片> --sex boy
```

测试：

```bash
python -m pytest tests/ -v
```

## 架构与关键文件

### 共享核心包 `common/`

改类别、分数表、骨龄公式、关节映射时**只改这里**：

- `common/constants.py`：
  - `DETECT_CLASSES`：YOLO 的 7 个骨骼类别，**顺序必须与 `configs/bone_detect.yaml` 的 `names` 一致**。
  - `CATEGORY`：13 个目标关节的输出顺序，**必须与 `common/bone_filter.py::filter_joints` 的输出顺序一致**。
  - `ARTHROSIS`：13 个关节 → 9 个分类模型（模型名 + 分类数）的映射。
  - `SCORE`：boy/girl × 关节 × 等级的分数表（RUS-CHN 计分法）。
- `common/bone_filter.py`：`filter_joints(boxes)` 把 YOLO 的 21 个框筛选/排序成 13 个目标关节（按 x 坐标排序取指定指骨）。
- `common/bone_age.py`：`calc_bone_age(score, sex)` 由总得分拟合骨龄的多项式（男女不同）；`export_report(...)` 生成中文结论文本。
- `common/image.py`：`trans_square` 等比补黑边成正方形；`clahe_detect`（整图检测前）、`clahe_crop`（关节分类前）的 CLAHE 参数。

### 数据处理 `data_prepare/`

- `voc_to_yolo.py`：VOC XML 标注 → YOLO txt 标签。
- `split_dataset.py`：按比例划分 train/val/test 并整理成 `images/{split}` 与 `labels/{split}`。
- `prepare_classify.py`：分类数据划分，每个关节目录下生成 `train.txt` / `val.txt`（每行 `图片路径 等级`）。

### 训练 `train/`

- `train/detect/train_detect.py`：`ultralytics` 检测训练入口。
- `train/classify/train_classify.py`：ResNet18（输入改单通道灰度、输出改为该关节级数）训练循环，AdamW，按 val acc 保存 `{category}_best.pth`。
- `train/classify/dataset.py`：`My_data` Dataset，读 `{split}.txt`（每行 `图片路径 标签`）。

### 推理 `inference/`

- `detector.py`：`load_detector()` 加载 YOLO 检测权重，`detect(model, img_path, conf)` 输出 `(img, boxes)`；`Detector` 类把模型包成 `.detect(img_path, conf)` 接口。
- `classifier.py`：`load_classifiers()` 加载 9 个 ResNet18 分类器；`classify_crop(model, pil_img)` 对单个关节裁剪图分级。
- `pipeline.py`：`predict_bone_age(...)` 端到端：检测 → 筛 13 关节 → 裁剪分级 → 查分表 → 算骨龄。
- `predict.py`：CLI 入口 `python -m inference.predict --img <图> --sex boy|girl [--conf 0.7] [--weights ...] [--classify-dir ...]`。

## 数据流约定

- YOLO 类别顺序由 `configs/bone_detect.yaml` 的 `names` 定义，与 `common/constants.py::DETECT_CLASSES` 保持一致：
  `0=Radius, 1=Ulna, 2=MCPFirst, 3=MCP, 4=ProximalPhalanx, 5=MiddlePhalanx, 6=DistalPhalanx`。
- `filter_joints` 依据该类别号 + 框的 x 坐标排序挑选关节，因此**类别号顺序与排序位置强耦合**，改动其一需同步改动另一处。

## 注意事项

- **类别顺序强耦合**：`configs/bone_detect.yaml`、`common/constants.py::DETECT_CLASSES`、`common/bone_filter.py` 三处必须同步修改。
- **框数校验**：检测输出框数不为 21 时 `filter_joints` 抛 `ValueError`，`inference/predict.py` 捕获后打印友好报错并以退出码 1 结束（不是 traceback）。
- **权重缺失校验**：检测权重与分类权重缺失时分别抛 `FileNotFoundError`，提示先训练或下载。
- **外部依赖未随仓库提供**：`models/`（权重）与 `datasets/`（数据）都在 `.gitignore` 中，需自行下载/准备。当前检测权重是 COCO 版 `models/yolov5su.pt`，跑骨龄推理前需换成用 `configs/bone_detect.yaml` 训练的 7 类骨骼检测权重。
- **环境**：torch 2.6.0 + cu126（GPU 版），安装见 README；Win 上默认装的是 CPU 版，需按官网命令换 GPU 版。
- 分类模型推理时对裁剪图做了 CLAHE 增强（`inference/classifier.py` 里 clipLimit=2.0, tileGridSize=(3,3)），训练/数据准备时用的是相同参数，两者需保持一致。
