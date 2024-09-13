# -*- coding: utf-8 -*-
import sys

import cv2
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage

import Window

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog


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
        self.ui.recButton.clicked.connect(self.openRecWindow)
        self.ui.mngButton.clicked.connect(self.openMngWindow)

    def updateFrame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)

            if qImg.isNull():
                print("Warning: QImage is null")
            else:
                pixmap = QPixmap.fromImage(qImg)
                scaled_pixmap = pixmap.scaled(self.ui.camLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.ui.camLabel.setPixmap(scaled_pixmap)
        else:
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

