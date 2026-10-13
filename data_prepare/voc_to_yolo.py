import argparse
import os
import xml.etree.ElementTree as ET

from tqdm import tqdm


def _convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (round(x * dw, 6), round(y * dh, 6), round(w * dw, 6), round(h * dh, 6))


def voc_to_yolo(xml_dir, out_dir, classes):
    os.makedirs(out_dir, exist_ok=True)
    names = [n for n in os.listdir(xml_dir) if n.endswith(".xml")]
    count = 0
    for name in tqdm(names):
        image_id = name[:-4]
        tree = ET.parse(os.path.join(xml_dir, name))
        root = tree.getroot()
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        lines = []
        for obj in root.iter("object"):
            if int(obj.find("difficult").text) == 1:
                continue
            cls = obj.find("name").text
            if cls not in classes:
                continue
            b = obj.find("bndbox")
            xmin = float(b.find("xmin").text)
            xmax = float(b.find("xmax").text)
            ymin = float(b.find("ymin").text)
            ymax = float(b.find("ymax").text)
            xmax = min(xmax, w)
            ymax = min(ymax, h)
            cx, cy, bw, bh = _convert((w, h), (xmin, xmax, ymin, ymax))
            lines.append("{} {} {} {} {}".format(classes.index(cls), cx, cy, bw, bh))
        with open(os.path.join(out_dir, image_id + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="VOC XML 标注转 YOLO txt")
    parser.add_argument("--xml-dir", required=True, help="VOC Annotations 目录")
    parser.add_argument("--out-dir", required=True, help="YOLO labels 输出目录")
    parser.add_argument("--classes", required=True, help="逗号分隔的类别名，顺序即类别 id")
    args = parser.parse_args()
    n = voc_to_yolo(args.xml_dir, args.out_dir, args.classes.split(","))
    print("转换完成，共 {} 个文件".format(n))


if __name__ == "__main__":
    main()
