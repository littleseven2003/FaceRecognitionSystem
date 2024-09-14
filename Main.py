# -*- coding: utf-8 -*-
import os
import shutil
import sys
import cv2
import cv2
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage

import Window

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

Cap_index = 0 # 0表示默认摄像头
current_path = os.getcwd()
data_path = current_path + "/data"
get_picture_num = 50 # 录入时拍摄照片数量
Total_data_num = 0 # 人脸数据总数
camera = None
data_manager = None
class DataDirectoryManager:
    def __init__(self):
        # 检查 data 文件夹是否存在
        if not os.path.exists(data_path):
            # 如果 data 文件夹不存在，则创建它
            os.makedirs(data_path)

        self.create_data_file()
    def create_data_file(self):
        global Total_data_num

        # 统计 data 文件夹下的子文件夹
        subfolders = [f.name for f in os.scandir(data_path) if f.is_dir()]
        num_subfolders = len(subfolders)

        # 创建或覆盖 .data 文件
        data_file_path = os.path.join(data_path, '.data')
        with open(data_file_path, 'w', encoding='utf-8') as f:
            f.write(f"{num_subfolders}\n")
            for subfolder in subfolders:
                f.write(f"{subfolder}\n")

        # 读取 .data 文件的第一行并将其写入全局变量
        with open(data_file_path, 'r', encoding='utf-8') as f:
            Total_data_num = int(f.readline().strip())
    def tmpPicDir(self):
        tmp_pic_path = os.path.join(data_path, "tmp_pic")
        if not os.path.exists(tmp_pic_path):
            os.mkdir(tmp_pic_path)
        else:
            shutil.rmtree(tmp_pic_path)
            os.mkdir(tmp_pic_path)

    def writeTmpPicDir(self, picName, picId):
        tmp_pic_path = os.path.join(data_path, "tmp_pic")
        new_folder_name = f"{picName}_{picId}"
        new_folder_path = os.path.join(data_path, new_folder_name)

        if os.path.exists(tmp_pic_path):
            os.rename(tmp_pic_path, new_folder_path)
        else:
            raise FileNotFoundError(f"The folder {tmp_pic_path} does not exist.")

        self.create_data_file()


class Camera:
    def __init__(self, Cap_index=0):
        # 初始化摄像头
        self.cap = cv2.VideoCapture(Cap_index)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")

        # 使用 Haar 级联分类器进行人脸检测
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def read_frame(self):
        # 从摄像头读取一帧图像
        ret, frame = self.cap.read()
        if ret:
            # 将图像从 BGR 转换为 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        else:
            raise Exception("Could not read frame from camera")

    def gray_img(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return gray

    def detect_faces(self, gray, frame):
        # 使用 Haar 级联分类器检测人脸
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(150, 150)
        )
        return faces

    def get_faces(self, gray, faces, index):
        #gray = self.gray_img(frame)
        #faces = self.detect_faces(gray, frame)
        status = 0
        for (x, y, w, h) in faces:
            # 保存图像，把灰度图片看成二维数组来检测人脸区域，这里是保存在data缓冲文件夹内
            cv2.imwrite(f"./data/tmp_pic/User.{index}.jpg", gray[y:y + h, x:x + w])
            status = 1
        return status



    def release(self):
        # 释放摄像头资源
        self.cap.release()




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

        self.okButton.clicked.connect(self.inputName)

    def inputName(self):
        self.picName = self.nameEdit.text()
        self.picId = self.idEdit.text()
        data_manager.writeTmpPicDir(self.picName, self.picId)
        self.close()


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

        self.okButton.clicked.connect(self.close)  # type: ignore
        # self.ui.okButton.clicked.connect(self.close)

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
        self.backButton.clicked.connect(self.close)

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Window.Ui_MainWindow()
        self.ui.setupUi(self)

        # 设置窗口全屏显示
        self.showFullScreen()

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
        # 连接 exitButton 的点击信号到槽函数
        self.ui.exitButton.clicked.connect(self.close)

    def updateFrame(self):
        try:
            frame = camera.read_frame()
            gray = camera.gray_img(frame)
            faces = camera.detect_faces(gray, frame)

            # 在图像中绘制略大于人脸的矩形框（蓝色边框）
            for (x, y, w, h) in faces:
                padding = 20  # 调整这个值以改变边框的大小
                x1 = max(x - padding, 0)
                y1 = max(y - padding, 0)
                x2 = min(x + w + padding, frame.shape[1])
                y2 = min(y + h + padding, frame.shape[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 将 OpenCV 图像转换为 QImage
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)

            if qImg.isNull():
                print("Warning: QImage is null")
            else:
                pixmap = QPixmap.fromImage(qImg)
                scaled_pixmap = pixmap.scaled(self.ui.camLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.ui.camLabel.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error: {e}")
            self.timer.stop()

    def capture_images(self):
        # 设置状态为拍照中
        self.ui.loadingLabel.setText("状态：拍照中")
        self.ui.loadingBar.setValue(0)

        data_manager.tmpPicDir()

        try:
            index = 0
            while(index < get_picture_num) :
                frame = camera.read_frame()
                gray = camera.gray_img(frame)
                faces = camera.detect_faces(gray, frame)

                # 在图像中绘制略大于人脸的矩形框（蓝色边框）
                for (x, y, w, h) in faces:
                    padding = 20  # 调整这个值以改变边框的大小
                    x1 = max(x - padding, 0)
                    y1 = max(y - padding, 0)
                    x2 = min(x + w + padding, frame.shape[1])
                    y2 = min(y + h + padding, frame.shape[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # 保存检测到的人脸
                status = camera.get_faces(gray, faces, index)
                if status:
                    index += 1

                # 更新摄像头画面
                height, width, channel = frame.shape
                bytesPerLine = 3 * width
                qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qImg)
                scaled_pixmap = pixmap.scaled(self.ui.camLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.ui.camLabel.setPixmap(scaled_pixmap)

                # 更新进度条
                self.ui.loadingBar.setValue(int((index / get_picture_num) * 100))
                QApplication.processEvents()  # 保持UI更新

            # 拍照完成
            self.ui.loadingLabel.setText("状态：等待操作")
            self.ui.loadingBar.setValue(0)
            self.ent_window = EntWindow(self)
            self.ent_window.exec_()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            # 恢复 updateFrame 调用
            self.timer.start(20)

    def closeEvent(self, event):
        camera.release()
        super(MainWindow, self).closeEvent(event)

    def openRecWindow(self):
        self.rec_window = RecWindow(self)
        self.rec_window.exec_()

    def openEntWindow(self):
        self.timer.stop()  # 停止 updateFrame

        self.capture_images()  # 调用拍照方法
        #self.ent_window = EntWindow(self)
        #self.ent_window.exec_()

    def openMngWindow(self):
        self.rec_window = MngWindow(self)
        self.rec_window.exec_()









if __name__ == "__main__":

    # 创建 DataDirectoryManager 的实例，自动调用 __init__ 方法
    data_manager = DataDirectoryManager()
    camera = Camera()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

