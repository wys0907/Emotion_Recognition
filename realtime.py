from deepface import DeepFace
import cv2
import numpy as np

# 分析人脸
# img_path = "qqemo/1.jpeg"
# objs = DeepFace.analyze(img_path=img_path, actions=['age', 'gender', 'race', 'emotion'])
DeepFace.stream("database")