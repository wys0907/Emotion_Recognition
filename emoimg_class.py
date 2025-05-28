from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import os
# 这个是做图片表情分类的，修改要处理的文件夹路径，就可以处理所有图片，并做了保存
# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# 设置要处理的图片文件夹
images_folder = 'qqemo'  # 修改为您的图片文件夹路径
results_folder = 'results'  # 结果保存的文件夹

# 创建结果文件夹（如果不存在）
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# 尝试加载表情符号图片（可选）
feelings_faces = []
for index, emotion in enumerate(EMOTIONS):
    try:
        feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))
    except:
        print(f"无法加载表情图片: emojis/{emotion}.png")
        feelings_faces.append(None)


def process_image(image_path):
    """处理单个图片文件"""
    print(f"正在处理图片: {image_path}")

    # 读取图片
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"无法读取图片: {image_path}")
        return None

    # 调整图片大小
    frame = imutils.resize(frame, width=500)

    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_detection.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # 创建一个画布来显示情绪概率
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()

    # 如果检测到人脸
    if len(faces) > 0:
        # 按照人脸面积排序，选择最大的人脸
        faces = sorted(faces, reverse=True,
                       key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))

        for i, face_coords in enumerate(faces):
            # 只处理最大的人脸
            if i > 0:
                break

            (fX, fY, fW, fH) = face_coords

            # 提取人脸区域
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # 进行情绪分类预测
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]

            # 在画布上绘制每种情绪的概率条
            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # 构建标签文本
                text = "{}: {:.2f}%".format(emotion, prob * 100)

                # 绘制概率条
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5),
                              (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 2)

            # 在人脸上标注检测到的情绪
            cv2.putText(frameClone, label, (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                          (0, 0, 255), 2)

            # 如果有表情符号图片，添加到图像上（可选）
            emoji_face = feelings_faces[preds.argmax()]
            if emoji_face is not None:
                # 确保emoji图片有alpha通道
                if emoji_face.shape[2] == 4:
                    # 在图片的指定位置添加表情符号
                    for c in range(0, 3):
                        height, width = emoji_face.shape[:2]
                        if fY + fH + 10 + height <= frameClone.shape[0] and fX + width <= frameClone.shape[1]:
                            frameClone[fY + fH + 10:fY + fH + 10 + height, fX:fX + width, c] = \
                                emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + \
                                frameClone[fY + fH + 10:fY + fH + 10 + height, fX:fX + width, c] * \
                                (1.0 - emoji_face[:, :, 3] / 255.0)

        return frameClone, canvas
    else:
        # 没有检测到人脸
        cv2.putText(frameClone, "No face detected", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frameClone, canvas


def main():
    """处理文件夹中的所有图片"""
    # 检查图片文件夹是否存在
    if not os.path.exists(images_folder):
        print(f"图片文件夹不存在: {images_folder}")
        return

    # 获取所有图片文件
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(images_folder) if
                   os.path.isfile(os.path.join(images_folder, f)) and
                   os.path.splitext(f.lower())[1] in valid_extensions]

    if not image_files:
        print(f"在文件夹中未找到图片: {images_folder}")
        return

    print(f"找到 {len(image_files)} 张图片，开始处理...")

    for img_file in image_files:
        image_path = os.path.join(images_folder, img_file)

        try:
            # 处理图片
            result = process_image(image_path)
            if result is None:
                continue

            frame_result, canvas = result

            # 保存处理结果
            base_name = os.path.splitext(img_file)[0]
            result_path = os.path.join(results_folder, f"{base_name}_result.jpg")
            canvas_path = os.path.join(results_folder, f"{base_name}_probabilities.jpg")

            cv2.imwrite(result_path, frame_result)
            cv2.imwrite(canvas_path, canvas)
            print(f"已保存结果: {result_path}")

            # 显示结果
            cv2.imshow('Emotion Detection', frame_result)
            cv2.imshow('Probabilities', canvas)

            # 等待按键，按q退出
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
        except Exception as e:
            print(f"处理图片 {img_file} 时出错: {e}")

    # 关闭所有窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()