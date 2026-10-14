import argparse
import sys

from inference.classifier import load_classifiers
from inference.detector import Detector
from inference.pipeline import predict_bone_age


def _run(img_path, sex, conf, weights, classify_dir):
    try:
        detector = Detector(weights)
        classifiers = load_classifiers(classify_dir)
        report = predict_bone_age(img_path, sex, detector, classifiers, conf=conf)
    except ValueError as e:
        print("错误: {}".format(e), file=sys.stderr)
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
