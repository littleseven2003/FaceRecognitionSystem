# -*- coding: utf-8 -*-
import sys

import cv2
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage

import Window

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



class EntWindow(QDialog, Window.Ui_EntWindow):
    def __init__(self, parent=None):
        super(EntWindow, self).__init__(parent)

        # 获取当前的窗口标志
        #current_flags = self.windowFlags()
        # 设置禁用最大化、最小化和关闭按钮
        #self.setWindowFlags(
        #    current_flags & ~QtCore.Qt.WindowMinimizeButtonHint & ~QtCore.Qt.WindowMaximizeButtonHint & ~QtCore.Qt.WindowCloseButtonHint)

        # 设置无标题栏窗口
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)

        self.setupUi(self)
        self.setWindowFlag(QtCore.Qt.Dialog)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)

class RecWindow(QDialog, Window.Ui_RecWindow):
    def __init__(self, parent=None):
        super(RecWindow, self).__init__(parent)


        # 获取当前的窗口标志
        #current_flags = self.windowFlags()
        # 设置禁用最大化、最小化和关闭按钮
        #self.setWindowFlags(
        #    current_flags & ~QtCore.Qt.WindowMinimizeButtonHint & ~QtCore.Qt.WindowMaximizeButtonHint & ~QtCore.Qt.WindowCloseButtonHint)

        # 设置无标题栏窗口
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)

        self.setupUi(self)
        self.setWindowFlag(QtCore.Qt.Dialog)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)

class MngWindow(QDialog, Window.Ui_MngWindow):
    def __init__(self, parent=None):
        super(MngWindow, self).__init__(parent)
        self.showFullScreen()  # 设置窗口全屏显示

        # 获取当前的窗口标志
        #current_flags = self.windowFlags()
        # 设置禁用最大化、最小化和关闭按钮
        #self.setWindowFlags(
        #    current_flags & ~QtCore.Qt.WindowMinimizeButtonHint & ~QtCore.Qt.WindowMaximizeButtonHint & ~QtCore.Qt.WindowCloseButtonHint)

        # 设置无标题栏窗口
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)


        self.setupUi(self)
        self.setWindowFlag(QtCore.Qt.Dialog)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Window.Ui_MainWindow()
        self.ui.setupUi(self)

        # 设置窗口全屏显示
        self.showFullScreen()

        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)  # 0表示默认摄像头
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(20)  # 每20毫秒更新一次

        # 设置 cameraWidget 风格
        self.ui.camLabel.setStyleSheet("border: 2px solid black; background-color: black;")

        # 连接 entButton 的点击信号到槽函数
        self.ui.entButton.clicked.connect(self.openEntWindow)
        # 连接 recButton 的点击信号到槽函数
        self.ui.recButton.clicked.connect(self.openRecWindow)
        # 连接 mngButton 的点击信号到槽函数
        self.ui.mngButton.clicked.connect(self.openMngWindow)

    def updateFrame(self):
        # 从摄像头读取一帧图像
        ret, frame = self.cap.read()
        if ret:
            # 将图像从 BGR 转换为 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 将图像从 RGB 转换为灰度图像，用于人脸检测
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # 使用 Haar 级联分类器检测人脸
            faces = face_cascade.detectMultiScale(
                gray,  # 灰度图像
                scaleFactor=1.1,  # 图像缩放比例，每次图像尺寸减小 10% 。 较小的值（如 1.05）将使得检测更精细，但会增加计算时间。较大的值（如 1.4）将减少计算时间，但可能会错过一些人脸。
                minNeighbors=5,  # 每一个候选矩形需要有 5 个邻近的矩形才被保留。 如果误报较多，可以增加此值；如果漏检较多，可以减少此值。
                minSize=(150, 150)  # 最小检测的矩形大小。 根据图像分辨率和人脸大小调整。如果要检测较远的小人脸，可以减小此值；如果要忽略这些小人脸，可以增加此值。
                # maxSize=(300, 300) # 最大检测的矩形大小。 根据实际需求设置。如果要忽略大于一定尺寸的区域，可以设置此值。
            )

            # 在图像中绘制略大于人脸的矩形框（蓝色边框）
            for (x, y, w, h) in faces:
                # 调整边框大小，使其略大于人脸
                padding = 20  # 调整这个值以改变边框的大小
                # 计算新的边框坐标，确保不超过图像边界
                x1 = max(x - padding, 0)
                y1 = max(y - padding, 0)
                x2 = min(x + w + padding, frame.shape[1])
                y2 = min(y + h + padding, frame.shape[0])
                # 画出矩形框，颜色为蓝色 (0, 0, 255)，边框宽度为 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 获取图像的高度、宽度和通道数
            height, width, channel = frame.shape
            # 计算每行像素所占的字节数
            bytesPerLine = 3 * width
            # 将 OpenCV 图像转换为 QImage
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)

            # 检查 QImage 是否为空
            if qImg.isNull():
                print("Warning: QImage is null")
            else:
                # 将 QImage 转换为 QPixmap
                pixmap = QPixmap.fromImage(qImg)
                # 缩放 QPixmap 以适应标签大小，并保持纵横比
                scaled_pixmap = pixmap.scaled(self.ui.camLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                # 在标签上显示缩放后的 QPixmap
                self.ui.camLabel.setPixmap(scaled_pixmap)
        else:
            # 如果无法从摄像头读取帧，则打印错误信息并停止定时器
            print("Error: Could not read frame from camera.")
            self.timer.stop()

    def closeEvent(self, event):
        self.cap.release()
        super(MainWindow, self).closeEvent(event)


    def openRecWindow(self):
        self.rec_window = RecWindow(self)
        self.rec_window.exec_()  # 使用 exec_() 以确保 MainWindow 被禁用，EntWindow 始终在最上层

    def openEntWindow(self):
        self.ent_window = EntWindow(self)
        self.ent_window.exec_()  # 使用 exec_() 以确保 MainWindow 被禁用，EntWindow 始终在最上层

    def openMngWindow(self):
        self.rec_window = MngWindow(self)
        self.rec_window.exec_()  # 使用 exec_() 以确保 MainWindow 被禁用，EntWindow 始终在最上层

if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

