# 骨龄检测

手部 X 光片 → 检测 21 个骨骼框 → 筛选 13 个目标关节 → 分类分级 → RUS-CHN 计分 → 骨龄。

## 环境

- Python 3.11
- `pip install -r requirements.txt`
- GPU 版 torch：`pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126`

## 模块

### ① 数据处理
```bash
python -m data_prepare.voc_to_yolo --xml-dir <VOC Annotations> --out-dir <labels> --classes Radius,Ulna,MCPFirst,MCP,ProximalPhalanx,MiddlePhalanx,DistalPhalanx
python -m data_prepare.split_dataset --images-dir <JPEGImages> --labels-dir <labels> --out-root <VOCdevkit>
python -m data_prepare.prepare_classify --data-root <关节数据根目录>
```

### ② 训练
```bash
python -m train.detect.train_detect --data configs/bone_detect.yaml --weights models/yolov5su.pt
python -m train.classify.train_classify --data-root <分类数据根目录>
```

### ③ 推理
```bash
python -m inference.predict --img <X光片> --sex boy
```

## 测试
```bash
python -m pytest tests/ -v
```
