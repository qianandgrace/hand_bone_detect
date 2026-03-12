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
move mydata.yaml 到yolov5的data路径下
python train.py --batch 16 --epochs 300 --data mydata.yaml --weights yolov5s.pt
Fusing layers... 
Model summary: 157 layers, 7029004 parameters, 0 gradients, 15.8 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 8/8 
                   all        238       4994      0.993      0.991      0.989      0.597
                Radius        238        243      0.992      0.971      0.975      0.676
                  Ulna        238        237      0.992      0.994      0.994      0.613
              MCPFirst        238        237      0.987       0.98      0.983      0.543
                   MCP        238        944      0.991      0.998      0.992      0.573
       ProximalPhalanx        238       1195      0.996      0.992      0.994      0.602
         MiddlePhalanx        238        948      0.995          1      0.991      0.557
         DistalPhalanx        238       1190          1          1      0.995      0.615
Results saved to runs\train\exp6

python hand_test.py 训练分类模型

# 部署
flask 部署，可以通过上传手部x光图片判断年龄
cd hand_bone_detect
## 单张图片检测
python detect_bone.py

## 服务拉起
python flask test_img.py
效果如下图
