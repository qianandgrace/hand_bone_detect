"""骨龄计算与中文报告生成。

从 hand_bone_detect/common.py 的 calcBoneAge / export 迁移而来，算法与文案保持不变。
"""


def calc_bone_age(score, sex):
    """根据 RUS-CHN 总得分与性别，用多项式拟合公式计算骨龄（岁）。"""
    if sex == "boy":
        bone_age = (
            2.01790023656577
            - 0.0931820870747269 * score
            + score ** 2 * 0.00334709095418796
            + score ** 3 * (-3.32988302362153e-05)
            + score ** 4 * (1.75712910819776e-07)
            + score ** 5 * (-5.59998691223273e-10)
            + score ** 6 * (1.1296711294933e-12)
            + score ** 7 * (-1.45218037113138e-15)
            + score ** 8 * (1.15333377080353e-18)
            + score ** 9 * (-5.15887481551927e-22)
            + score ** 10 * (9.94098428102335e-26)
        )
        return round(bone_age, 2)
    if sex == "girl":
        bone_age = (
            5.81191794824917
            - 0.271546561737745 * score
            + score ** 2 * 0.00526301486340724
            + score ** 3 * (-4.37797717401925e-05)
            + score ** 4 * (2.0858722025667e-07)
            + score ** 5 * (-6.21879866563429e-10)
            + score ** 6 * (1.19909931745368e-12)
            + score ** 7 * (-1.49462900826936e-15)
            + score ** 8 * (1.162435538672e-18)
            + score ** 9 * (-5.12713017846218e-22)
            + score ** 10 * (9.78989966891478e-26)
        )
        return round(bone_age, 2)
    raise ValueError("sex 必须是 'boy' 或 'girl'，收到: {!r}".format(sex))


def export_report(results, score, bone_age):
    """生成 13 关节分级明细 + 总分与骨龄的中文报告文本。"""
    return """
    第一掌骨骺分级{}级，得{}分；第三掌骨骨骺分级{}级，得{}分；第五掌骨骨骺分级{}级，得{}分；
    第一近节指骨骨骺分级{}级，得{}分；第三近节指骨骨骺分级{}级，得{}分；第五近节指骨骨骺分级{}级，得{}分；
    第三中节指骨骨骺分级{}级，得{}分；第五中节指骨骨骺分级{}级，得{}分；
    第一远节指骨骨骺分级{}级，得{}分；第三远节指骨骨骺分级{}级，得{}分；第五远节指骨骨骺分级{}级，得{}分；
    尺骨分级{}级，得{}分；桡骨骨骺分级{}级，得{}分。

    RUS-CHN分级计分法，受检儿CHN总得分：{}分，骨龄约为{}岁。""".format(
        results['MCPFirst'][0], results['MCPFirst'][1],
        results['MCPThird'][0], results['MCPThird'][1],
        results['MCPFifth'][0], results['MCPFifth'][1],
        results['PIPFirst'][0], results['PIPFirst'][1],
        results['PIPThird'][0], results['PIPThird'][1],
        results['PIPFifth'][0], results['PIPFifth'][1],
        results['MIPThird'][0], results['MIPThird'][1],
        results['MIPFifth'][0], results['MIPFifth'][1],
        results['DIPFirst'][0], results['DIPFirst'][1],
        results['DIPThird'][0], results['DIPThird'][1],
        results['DIPFifth'][0], results['DIPFifth'][1],
        results['Ulna'][0], results['Ulna'][1],
        results['Radius'][0], results['Radius'][1],
        score, bone_age,
    )
