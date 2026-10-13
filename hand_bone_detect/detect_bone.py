import os

from detect_utils import load_detector, detect_only

if __name__ == "__main__":
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data", "example.jpg")

    model = load_detector()
    print("检测模型加载完成，开始检测:", img_path)

    result = detect_only(model, img_path, conf=0.25)
    boxes = result.boxes
    print("检测到目标数量:", len(boxes))
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
        print("  类别={} 置信度={:.3f} 坐标={}".format(model.names[cls_id], conf, xyxy))
