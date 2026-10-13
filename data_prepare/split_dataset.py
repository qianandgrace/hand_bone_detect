import argparse
import os
import random
import shutil


def split_dataset(images_dir, labels_dir, out_root, trainval=0.9, train=0.7, seed=0):
    """按比例划分并拷贝成 images/{train,val,test} 与 labels/{train,val,test}。"""
    random.seed(seed)
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(images_dir))
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
