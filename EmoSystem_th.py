import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from EmoR import Ui_Dialog
from EmoAna import Analysis
import traceback


class AnalysisWorker(QThread):
    """工作线程类，处理耗时的分析工作"""
    # 定义信号
    analysisComplete = pyqtSignal(str, str, str,
                                  str)  # 分析完成信号(summary_text, emotion_text, redbox_path, combined_img_path)
    analysisError = pyqtSignal(str)  # 错误信号

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        """线程执行的主要任务"""
        try:
            # 在这里调用分析函数
            summary_text, emotion_text, redbox_path, combined_img_path = Analysis(self.image_path)
            self.analysisComplete.emit(summary_text, emotion_text, redbox_path, combined_img_path)
        except Exception as e:
            error_info = f"分析过程出错: {str(e)}\n{traceback.format_exc()}"
            self.analysisError.emit(error_info)


class EmotionDetectionApp(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("人脸表情识别分类系统")

        # 初始化变量
        self.or_emo_path = ""
        self.combined_img_path = ""
        self.worker = None

        # 连接按钮到函数
        self.pushButton.clicked.connect(self.upload_image)
        self.pushButton_2.clicked.connect(self.start_detection)
        self.pushButton_3.clicked.connect(self.save_result)

        # 设置错误处理
        sys.excepthook = self.handle_exception

    def handle_exception(self, exc_type, exc_value, exc_traceback):
        """全局异常处理器"""
        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        self.show_error_message("程序错误", f"发生未捕获的异常:\n{error_msg}")
        # 打印到控制台以便调试
        print(f"未捕获的异常: {error_msg}")

    def show_error_message(self, title, message):
        """显示错误消息对话框"""
        QMessageBox.critical(self, title, message)

    def upload_image(self):
        """打开文件对话框并加载选定的图像"""
        try:
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
        except Exception as e:
            self.show_error_message("上传图片错误", f"上传图片时发生错误: {str(e)}")

    def display_image(self, image_path, graphics_view):
        """在指定的图像视图中显示图像"""
        try:
            pixmap = QPixmap(image_path)
            scene = QGraphicsScene()
            scene.addPixmap(pixmap)
            graphics_view.setScene(scene)
            graphics_view.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        except Exception as e:
            self.show_error_message("显示图像错误", f"显示图像时发生错误: {str(e)}")

    def start_detection(self):
        """启动情绪检测过程"""
        if not self.or_emo_path:
            self.show_error_message("错误", "请先上传图片")
            return

        # 显示加载指示
        self.summary.setText("正在分析中，请稍候...")
        self.pushButton_2.setEnabled(False)

        # 创建并启动工作线程
        self.worker = AnalysisWorker(self.or_emo_path)
        self.worker.analysisComplete.connect(self.handle_analysis_complete)
        self.worker.analysisError.connect(self.handle_analysis_error)
        self.worker.finished.connect(lambda: self.pushButton_2.setEnabled(True))
        self.worker.start()

    @pyqtSlot(str, str, str, str)
    def handle_analysis_complete(self, summary_text, emotion_text, redbox_path, combined_img_path):
        """处理分析完成信号"""
        try:
            self.summary.setText(summary_text)
            self.text2.setText(emotion_text)
            self.combined_img_path = combined_img_path
            self.display_image(combined_img_path, self.combined_img)
        except Exception as e:
            self.show_error_message("结果显示错误", f"显示分析结果时发生错误: {str(e)}")

    @pyqtSlot(str)
    def handle_analysis_error(self, error_message):
        """处理分析错误信号"""
        self.summary.setText(f"分析过程中发生错误，请检查日志")
        self.show_error_message("分析错误", error_message)
        # 将错误信息写入日志文件
        try:
            with open("error_log.txt", "a") as log_file:
                log_file.write(f"\n--- {QtCore.QDateTime.currentDateTime().toString()} ---\n")
                log_file.write(error_message)
                log_file.write("\n---------------------------------\n")
        except:
            pass  # 如果日志写入失败，我们不希望引发另一个异常

    def save_result(self):
        """保存分析结果"""
        try:
            if not hasattr(self, 'combined_img_path') or not self.combined_img_path:
                self.show_error_message("保存错误", "没有可保存的结果，请先进行分析")
                return

            # 显示保存路径
            self.summary.append("--------------")
            self.summary.append(f"图片已保存到: {self.combined_img_path}")
            self.summary.append("--------------")
        except Exception as e:
            self.show_error_message("保存结果错误", f"保存结果时发生错误: {str(e)}")

    def resizeEvent(self, event):
        """窗口大小改变时保持图像纵横比"""
        super().resizeEvent(event)

        try:
            # 更新图像视图
            if hasattr(self, 'or_emo_path') and self.or_emo_path:
                if self.org_img.scene():
                    self.org_img.fitInView(self.org_img.scene().sceneRect(), QtCore.Qt.KeepAspectRatio)

            if hasattr(self, 'combined_img_path') and self.combined_img_path:
                if self.combined_img.scene():
                    self.combined_img.fitInView(self.combined_img.scene().sceneRect(), QtCore.Qt.KeepAspectRatio)
        except Exception as e:
            # 这里我们不显示消息框，因为它可能会导致更多的resizeEvent
            print(f"调整大小时出错: {str(e)}")

    def closeEvent(self, event):
        """窗口关闭时确保线程正确结束"""
        if self.worker and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait()
        event.accept()


if __name__ == "__main__":
    try:
        app = QtWidgets.QApplication(sys.argv)
        window = EmotionDetectionApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"程序启动错误: {str(e)}")
        print(traceback.format_exc())
        # 显示错误对话框
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("程序启动时发生严重错误")
        msg.setInformativeText(str(e))
        msg.setDetailedText(traceback.format_exc())
        msg.setWindowTitle("启动错误")
        msg.exec_()