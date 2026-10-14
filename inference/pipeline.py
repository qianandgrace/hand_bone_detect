from PIL import Image

from common.bone_age import calc_bone_age, export_report
from common.bone_filter import filter_joints
from common.constants import ARTHROSIS, CATEGORY, SCORE
from inference.classifier import classify_crop


def predict_bone_age(img_path, sex, detector, classifiers, conf=0.7):
    """端到端：检测 -> 筛 13 关节 -> 裁剪分级 -> 查分表 -> 算骨龄。"""
    img, boxes = detector.detect(img_path, conf=conf)
    joints = filter_joints(boxes).cpu().numpy()

    score_all = 0
    bone_results = {}
    for name, box in zip(CATEGORY, joints):
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        crop = img[y1:y2, x1:x2]
        crop = Image.fromarray(crop[:, :, ::-1].copy())  # BGR -> RGB
        model = classifiers[ARTHROSIS[name][0]]
        grade = classify_crop(model, crop)
        score = SCORE[sex][name][grade]
        bone_results[name] = [grade + 1, int(score)]
        score_all += score

    bone_age = calc_bone_age(score_all, sex)
    return export_report(bone_results, score_all, bone_age)
