# 骨龄检测项目重构设计

- 日期：2026-10-13
- 状态：待评审
- 范围：仓库结构重整为三大模块 + 共享核心包，并清理冗余代码

## 背景

项目基于手部 X 光片预测骨龄：检测关节 → 筛选目标关节 → 切割 → 分类分级 → 查分表 → 代入公式算骨龄。
原实现脚本散落根目录与两个子目录，职责交叉、路径硬编码、存在重复且不一致的实现。检测部分已从 `yolov5` 独立仓库迁移到 `ultralytics` 库（`yolov5su.pt` 加载已跑通），目标是把整个仓库重整为清晰的三模块结构。

## 现状问题

1. 数据处理脚本（`images_tag.py` / `voc_to_yolo.py` / `transfer_files.py`）散在根目录，且硬编码 `C:\Users\qian gao\...` 绝对路径。
2. 训练代码（`hand_test/`）与部署代码（`hand_bone_detect/`）目录名与实际职责不符。
3. `bone_filter` 有两份实现（`common.py` 与 `bone_filter_utils.py`），类别映射与输出顺序不一致，是真实隐患。
4. 部署目录夹带 Flask / PyQt5 前端与演示脚本，与"推理"职责混杂。
5. 分类权重 `.pth`（9 个，约 400MB）提交在 git 中，仓库臃肿。
6. 无任何测试。
7. 检测框数异常时只有一句 `print`，随后会以难以理解的方式崩溃。

## 目标结构

```
hand_bone_detect/
├─ common/                  # 共享领域逻辑（跨模块复用，单一来源）
│   ├─ __init__.py
│   ├─ constants.py         # DETECT_CLASSES(7检测类)、CATEGORY(13关节)、ARTHROSIS(关节→分类模型)、SCORE(分数表)
│   ├─ bone_filter.py       # 21框 → 13关节 的筛选排序
│   ├─ bone_age.py          # calcBoneAge、export_report
│   └─ image.py             # trans_square、CLAHE 预处理
├─ data_prepare/            # ① 数据处理
│   ├─ voc_to_yolo.py       # VOC XML → YOLO txt
│   ├─ split_dataset.py     # 划分 train/val/test + 拷贝到 images/ 与 labels/
│   └─ prepare_classify.py  # 分类数据：切分 train/val、生成 txt、可选增样
├─ train/                   # ② 训练
│   ├─ detect/train_detect.py      # ultralytics YOLO 训练
│   └─ classify/
│       ├─ dataset.py              # My_data Dataset
│       └─ train_classify.py       # 9 个 ResNet18 分类模型
├─ inference/               # ③ 推理（开发用）
│   ├─ detector.py          # 加载检测模型 + 检测
│   ├─ classifier.py        # 加载分类模型 + 分级
│   ├─ pipeline.py          # 端到端：检测→裁剪→分级→得分→骨龄
│   └─ predict.py           # CLI 入口
├─ configs/
│   └─ bone_detect.yaml     # 检测数据集配置（ultralytics 格式）
├─ models/                  # 权重（gitignore）
│   ├─ detect/              # 检测权重
│   └─ classify/            # 9 个分类权重
├─ tests/                   # 最小单测
├─ requirements.txt
└─ README.md
```

内层 `hand_bone_detect/hand_bone_detect/` 目录消失，内容并入 `inference/` 与 `common/`。

## 模块职责与接口

### common/（纯逻辑，不加载权重）

- `constants.py`：`DETECT_CLASSES`（7 检测类，顺序须与检测数据集一致）、`CATEGORY`（13 关节输出顺序）、`ARTHROSIS`（13 关节 → 9 分类模型的映射）、`SCORE`（男/女分数表）。无外部依赖。
- `bone_filter.py`：`filter_joints(boxes) -> Tensor[13, 6]`，把 21 框按类别与 x 坐标筛选排序成 13 关节。**合并现有两份实现为唯一来源**，采用与 `configs/bone_detect.yaml` 类别顺序自洽的那份。
- `bone_age.py`：`calc_bone_age(score, sex)`、`export_report(results, score, bone_age)`。
- `image.py`：`trans_square(image)`、`to_clahe_gray(img)`（统一训练与推理的 CLAHE 参数）。

### data_prepare/（每个脚本一个 CLI，文件进文件出）

- `voc_to_yolo.py`：参数 `--xml-dir --out-dir --classes`，不再硬编码。
- `split_dataset.py`：参数 `--ratio`，划分 train/val/test 并拷贝到 `images/` 与 `labels/`（合并现有 `images_tag.py` + `transfer_files.py`）。
- `prepare_classify.py`：切分分类数据、生成 `train.txt`/`val.txt`、可选去雾与旋转增样（源自 `hand_test/utils/data_utils.py`）。

### train/

- `detect/train_detect.py`：`from ultralytics import YOLO; YOLO('yolov5su.pt').train(data='configs/bone_detect.yaml', ...)`。
- `classify/dataset.py`：`My_data` Dataset（源自 `hand_test/utils/data_set.py`）。
- `classify/train_classify.py`：9 个 ResNet18（输入单通道、输出类别数按关节），逻辑源自 `hand_test/trainer.py`。

### inference/

- `detector.py`：`load_detector()`、`detect(img_path, conf)`。
- `classifier.py`：`load_classifiers()`、`classify_joints(crops)`。
- `pipeline.py`：`predict_bone_age(img_path, sex)`，编排全链路。
- `predict.py`：CLI `python inference/predict.py --img x.jpg --sex boy`。

## 数据流

- 检测：`VOC XML` →(voc_to_yolo)→ `YOLO txt` →(+split_dataset)→ `images/{train,val,test}` + `labels/{train,val,test}` →(train_detect)→ `models/detect/*.pt`
- 分类：`关节图` →(prepare_classify)→ `train.txt/val.txt` →(train_classify)→ `models/classify/*.pth`
- 推理：`图片` →(detector)→ 21 框 →(bone_filter)→ 13 关节 →裁剪→(classifier)→ 等级 →(SCORE)→ 总分 →(calcBoneAge)→ 骨龄报告

## 配置与路径

- 检测数据集配置归 `configs/bone_detect.yaml`（`path/train/val/names`），由 `mydata.yaml` 改名并整理。
- 所有脚本用 `argparse` 传路径/参数，默认值基于 `__file__` 推导仓库根，杜绝硬编码绝对路径。

## 运行约定

- 各模块目录含 `__init__.py`；脚本统一以模块方式从仓库根运行，确保 `import common` 可用：
  `python -m data_prepare.voc_to_yolo --xml-dir ... --out-dir ...`、`python -m inference.predict --img x.jpg --sex boy`
- `requirements.txt`：torch、torchvision、ultralytics、opencv-python、pillow、numpy、pyyaml、pytest。

## 错误处理

- 检测框数 ≠ 21：抛 `ValueError`，信息说明"检测到 N 个框，预期 21，请检查检测模型/阈值"。
- 权重文件不存在：明确提示"请先训练或下载对应权重"，而非底层 traceback。

## 测试

新增 `tests/`，只覆盖纯函数：

- `test_bone_filter.py`：构造 21 框（含各类别、乱序 x），断言筛选出的 13 关节顺序与类别正确。
- `test_bone_age.py`：已知总分 → 断言 `calc_bone_age` 输出。
- `test_image.py`：`trans_square` 输出为正方形且尺寸正确。

## 删除清单

- `hand_bone_detect/flask_test_img.py`、`main.py`、`bone_detect_ui.ui`、`templates/`
- `hand_bone_detect/test.py`
- `hand_bone_detect/bone_filter_utils.py`（并入 `common/`）
- `hand_bone_detect/img/`、`test_data/`、`cutpictures/`、`detect_result/`、`__pycache__/`
- 根目录 `res.mhtml`
- `hand_test/utils/tools.py`（与 `common/image.py` 重复，合并）

## 迁移清单（用 `git mv` 保留历史）

- 根 `images_tag.py` + `transfer_files.py` → `data_prepare/split_dataset.py`（合并重写）
- 根 `voc_to_yolo.py` → `data_prepare/voc_to_yolo.py`（改写去硬编码）
- `hand_test/trainer.py` → `train/classify/train_classify.py`
- `hand_test/utils/data_set.py` → `train/classify/dataset.py`
- `hand_test/utils/data_utils.py` → `data_prepare/prepare_classify.py`
- `hand_bone_detect/detect_utils.py` + `detect_bone.py` → 拆分进 `inference/`（`detector.py` / `classifier.py` / `pipeline.py` / `predict.py`）
- `hand_bone_detect/common.py` → 拆分进 `common/`（`constants.py` / `bone_filter.py` / `bone_age.py` / `image.py`）
- `mydata.yaml` → `configs/bone_detect.yaml`
- `hand_test/params/*.pth` → `models/classify/*.pth`（**并从 git 移除跟踪**）

## 非目标（本次不做）

- 不训练任何模型（本仓库暂无可用骨骼检测权重，训练需另开任务）。
- 不改动骨龄算法本身（分数表、公式、13 关节定义维持现状）。
- 不为 GUI / Web 服务重建前端（已确认删除）。

## 验收标准

1. 目录结构与本文一致，无遗留散落脚本。
2. `pytest tests/` 全绿。
3. `python inference/predict.py --img <示例> --sex boy` 能加载检测模型并在 GPU 上完成推理（在无骨骼权重时以"检测框数异常"的明确报错结束，而非 traceback）。
4. 无硬编码 `C:\Users\qian gao\...` / `D:/dataset/...` 绝对路径。
5. `git status` 中 `models/` 被忽略；`.pth` 从跟踪中移除。
