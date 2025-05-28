from deepface import DeepFace
import cv2
import numpy as np
import os


# 分析人脸
class deepAnalysis:
    def __init__(self):
        pass

    def Analysis(img_path):
        objs = DeepFace.analyze(img_path=img_path, actions=['age', 'gender', 'race', 'emotion'])
        print(objs)
        return objs

    # 读取原始图片
    def plot_Ana_img(img_path, objs):
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            exit()

        # 在人脸上绘制边框
        face_data = objs[0]
        region = face_data['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']

        # 创建原图的副本，以保留原图
        img_with_box = img.copy()
        cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色边框

        # 获取图片尺寸
        height, width = img.shape[:2]

        # 创建布局 - 左侧原图，右侧两列信息
        info_width = 400  # 信息区域宽度
        new_height = max(height, 400)  # 确保足够的高度
        combined_img = np.ones((new_height, width + info_width, 3), dtype=np.uint8) * 255  # 白色背景

        # 将带有人脸框的图放在左侧 - 确保尺寸匹配
        combined_img[:height, :width] = img_with_box

        # 提取需要显示的信息
        age = face_data['age']
        gender = face_data['dominant_gender']
        race = face_data['dominant_race']
        emotion = face_data['dominant_emotion']
        emotions = face_data['emotion']

        # 设置文本参数
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 0)  # 黑色
        thickness = 2
        line_height = 30

        # 绘制分隔线
        cv2.line(combined_img, (width, 0), (width, new_height), (200, 200, 200), 1)

        # 第一列：基本信息
        col1_x = width + 20
        y_pos = 40

        # 绘制标题
        cv2.putText(combined_img, "Basic Information", (col1_x, int(y_pos)), font, font_scale, color, thickness)
        y_pos += line_height * 1

        # 基本信息
        # cv2.putText(combined_img, f"Age: {age}", (col1_x, int(y_pos)), font, font_scale, color, thickness)
        # y_pos += line_height
        cv2.putText(combined_img, f"Gender: {gender}", (col1_x, int(y_pos)), font, font_scale, color, thickness)
        y_pos += line_height
        # cv2.putText(combined_img, f"Race: {race}", (col1_x, int(y_pos)), font, font_scale, color, thickness)
        # y_pos += line_height
        cv2.putText(combined_img, f"Emotion: {emotion}", (col1_x, int(y_pos)), font, font_scale, color, thickness)
        y_pos += line_height * 1.5

        # 第二列：情绪概率
        # 绘制标题
        cv2.putText(combined_img, "Emotion Probabilities", (col1_x, int(y_pos)), font, font_scale, color, thickness)
        y_pos += line_height * 1

        # 情绪概率列表
        for emo, prob in emotions.items():
            # 处理 np.float32 类型
            if hasattr(prob, 'item'):
                prob_value = prob.item()
            else:
                prob_value = prob

            cv2.putText(combined_img, f"{emo}: {prob_value:.2f}%", (col1_x, int(y_pos)), font, font_scale, color,
                        thickness)
            y_pos += line_height

        # 智能处理输出路径
        base_name, ext = os.path.splitext(img_path)
        output_path = f"{base_name}_analyzed{ext}"

        cv2.imwrite(output_path, combined_img)
        return combined_img, output_path

    def plot_red_box(or_img_path, objs):
        img = cv2.imread(or_img_path)
        if img is None:
            print(f"无法读取图片: {or_img_path}")
            exit()

        # 创建原图的副本
        img_with_box = img.copy()

        # 在人脸上绘制边框
        face_data = objs[0]
        region = face_data['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色边框

        # 智能处理输出路径
        base_name, ext = os.path.splitext(or_img_path)
        output_path = f"{base_name}_redbox{ext}"

        cv2.imwrite(output_path, img_with_box)
        return img_with_box, output_path

    def save_img(combined_img_path):
        print(f"分析结果已保存至: {combined_img_path}!")

    def parse_deepface_result(result_list):
        result = result_list[0]

        # 整理 summary 信息
        summary = {
            '年龄': result.get('age', 'N/A'),
            '性别': result.get('dominant_gender', 'N/A'),
            '人种': result.get('dominant_race', 'N/A'),
            '情感': result.get('dominant_emotion', 'N/A'),
        }
        summary_text = "Result:\n" + "\n".join([f"{key}: {value}" for key, value in summary.items()])

        # 整理情绪概率信息
        emotions = result.get('emotion', {})
        if emotions:
            emotion_lines = [f"{emotion.title():<10}: {float(prob):.2f}%" for emotion, prob in emotions.items()]
            emotion_text = "各情绪概率:\n" + "\n".join(emotion_lines)
        else:
            emotion_text = "Emotion Probabilities:\nN/A"

        return summary_text, emotion_text


def Analysis(or_img_path):
    objs = deepAnalysis.Analysis(or_img_path)
    summary, emotion_text = deepAnalysis.parse_deepface_result(objs)

    _, redboxoutput_path = deepAnalysis.plot_red_box(or_img_path, objs)
    _, combined_img_path = deepAnalysis.plot_Ana_img(or_img_path, objs)

    return summary, emotion_text, redboxoutput_path, combined_img_path


if __name__ == "__main__":
    or_emo_path = r"qqemo/emo2.png"
    summary, emotion_text, redboxoutput_path, combined_img_path = Analysis(or_emo_path)
