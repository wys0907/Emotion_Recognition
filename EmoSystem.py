import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene
from PyQt5.QtGui import QPixmap, QImage
from EmoR import Ui_Dialog
from EmoAna import Analysis




class EmotionDetectionApp(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self):
        super().__init__()

        self.setupUi(self)
        self.setWindowTitle("表情情绪检测系统")

        # Initialize variables
        self.or_emo_path = ""
        self.combined_img_path = ""

        # Connect buttons to functions
        self.pushButton.clicked.connect(self.upload_image)
        self.pushButton_2.clicked.connect(self.start_detection)
        self.pushButton_3.clicked.connect(self.save_result)

    def upload_image(self):
        """Open file dialog and load selected image"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图像文件 (*.png *.jpg *.jpeg *.bmp *.gif);;所有文件 (*)",
            options=options
        )

        if file_name:
            self.or_emo_path = file_name
            self.display_image(self.or_emo_path, self.org_img)
            self.summary.clear()
            self.text2.clear()
            scene = QGraphicsScene()
            self.combined_img.setScene(scene)

    def display_image(self, image_path, graphics_view):

        pixmap = QPixmap(image_path)
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        graphics_view.setScene(scene)
        graphics_view.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def start_detection(self):


        # Call your analysis function
        summary_text, emotion_text, redbox_path, self.combined_img_path = Analysis(self.or_emo_path)
        # Display results
        self.summary.setText(summary_text)
        self.text2.setText(emotion_text)
        self.display_image(self.combined_img_path, self.combined_img)



    def save_result(self):

        # Simply display the path in the summary text box
        self.summary.append("--------------")
        self.summary.append(f"图片已保存到: {self.combined_img_path}")
        self.summary.append("--------------")

    def resizeEvent(self, event):
        """Maintain aspect ratio when window is resized"""
        super().resizeEvent(event)

        # Update image views when resizing
        if hasattr(self, 'or_emo_path') and self.or_emo_path:
            if self.org_img.scene():
                self.org_img.fitInView(self.org_img.scene().sceneRect(), QtCore.Qt.KeepAspectRatio)

        if hasattr(self, 'combined_img_path') and self.combined_img_path:
            if self.combined_img.scene():
                self.combined_img.fitInView(self.combined_img.scene().sceneRect(), QtCore.Qt.KeepAspectRatio)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = EmotionDetectionApp()
    window.show()
    sys.exit(app.exec_())