import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from common.constants import ARTHROSIS
from train.classify.dataset import My_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAMES = list(dict.fromkeys(v[0] for v in ARTHROSIS.values()))
NUM_CLASSES = {v[0]: v[1] for v in ARTHROSIS.values()}


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_model(num_classes):
    net = models.resnet18()
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net.fc = nn.Linear(512, num_classes)
    return net.to(DEVICE)


def train_classify(category, data_root, out_dir, epochs=80, batch_size=60):
    num_classes = NUM_CLASSES[category]
    train_set = My_data(os.path.join(data_root, category), "train")
    val_set = My_data(os.path.join(data_root, category), "val")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    model = create_model(num_classes)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "{}_best.pth".format(category))

    loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters())
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for img, label in train_loader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            loss = loss_func(model(img), label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        model.eval()
        acc_sum, n = 0.0, 0
        with torch.no_grad():
            for img, label in val_loader:
                img, label = img.to(DEVICE), label.to(DEVICE)
                out = model(img)
                acc_sum += torch.mean(torch.eq(out.argmax(dim=1), label).float()).item()
                n += 1
        val_acc = acc_sum / max(n, 1)
        print("epoch={} train_loss={:.4f} val_acc={:.4f}".format(epoch, train_loss / max(len(train_loader), 1), val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt)
            print("保存模型:", ckpt)


def main():
    parser = argparse.ArgumentParser(description="训练 9 个关节分级分类模型")
    parser.add_argument("--data-root", required=True, help="分类数据根目录（各关节子目录）")
    parser.add_argument("--out-dir", default=os.path.join(_repo_root(), "models", "classify"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--category", help="只训练指定关节（默认全部）")
    args = parser.parse_args()
    categories = [args.category] if args.category else MODEL_NAMES
    for cat in categories:
        print("开始训练:", cat)
        train_classify(cat, args.data_root, args.out_dir, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
