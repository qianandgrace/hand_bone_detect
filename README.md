# 方案介绍
模拟医生的操作
检测关节 -> 位置筛选目标关节 -> 切割关节 -> 计算所有关节等级对应分数 -> 输入公式，计算年龄

# 环境安装
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
在win上此时默认安装的是cpu版本，根据官网命令安装对应的gpu版本
此时安装2.5.1版本
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 数据准备
2017年RSNA的骨龄预测比赛数据集，原始数据标注格式为voc格式
https://www.kaggle.com/datasets/kmader/rsna-bone-age 

voc转为yolo格式
python images_tag.py 划分训练集 验证集 测试集
python voc_to_yolo.py 转化为yolo标签

仿照coco格式更改数据存储形式
python transfer_file.py


# 训练
# Example: Train YOLOv5s on the COCO128 dataset for 3 epochs
python train.py --batch 16 --epochs 300 --data mydata.yaml --weights yolov5s.pt


# 部署
