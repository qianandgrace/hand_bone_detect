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
