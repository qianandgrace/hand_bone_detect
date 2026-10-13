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
    for joint in sorted(os.listdir(data_root)):
        joint_dir = os.path.join(data_root, joint)
        if not os.path.isdir(joint_dir):
            continue
        train_list, val_list = [], []
        for grade in sorted(os.listdir(joint_dir)):
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
