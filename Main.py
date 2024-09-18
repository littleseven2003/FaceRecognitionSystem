# -*- coding: utf-8 -*-
import os
import shutil
import sys
import time
import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap, QImage
import Window
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

Cap_index = 0  # 0 表示默认摄像头（通常是内置摄像头或第一个外接摄像头）
Path = os.getcwd()  # 获取当前工作目录路径
Data_Path = Path + "/data"  # 定义数据存储路径，将在当前工作目录下创建一个名为 "data" 的文件夹
Img_Num = 2  # 录入时拍摄照片数量，用于在录入新的人脸数据时拍摄指定数量的照片
Data_Num = 0  # 人脸数据总数，用于统计并管理已录入的人脸数据数量

camera = None
data_manager = None

class DataManager:

    def __init__(self):
        # 检查 data 文件夹是否存在
        if not os.path.exists(Data_Path):  # 如果 data 文件夹不存在
            # 如果 data 文件夹不存在，则创建它
            os.makedirs(Data_Path)  # 创建 data 文件夹

        self.updateData()  # 调用 updateData 方法更新数据

    def updateData(self):
        global Data_Num

        # 统计 data 文件夹下的子文件夹
        subfolders = [f.name for f in os.scandir(Data_Path) if f.is_dir()]
        num_subfolders = len(subfolders)

        # 创建或覆盖 .data 文件
        data_file_path = os.path.join(Data_Path, '.data')
        with open(data_file_path, 'w', encoding='utf-8') as f:
            f.write(f"{num_subfolders}\n")
            for subfolder in subfolders:
                subfolder_data_file_path = os.path.join(Data_Path, subfolder, f"{subfolder}.data")
                if os.path.exists(subfolder_data_file_path):
                    with open(subfolder_data_file_path, 'r', encoding='utf-8') as subfolder_data_file:
                        lines = subfolder_data_file.readlines()
                        if len(lines) >= 2:
                            # 读取前两行并用 "_" 连接
                            combined_line = f"{lines[0].strip()}_{lines[1].strip()}"
                            f.write(f"{combined_line}\n")

        # 读取 .data 文件的第一行并将其写入全局变量
        with open(data_file_path, 'r', encoding='utf-8') as f:
            Data_Num = int(f.readline().strip())
    
    def CreateTmpImgDir(self):
        tmpImgPath = os.path.join(Data_Path, "_tmp_img_")
        if not os.path.exists(tmpImgPath):
            os.mkdir(tmpImgPath)
        else:
            shutil.rmtree(tmpImgPath)
            os.mkdir(tmpImgPath)

    def WriteTmpPicDir(self, picName, picId):
        tmpImgPath = os.path.join(Data_Path, "_tmp_img_")
        new_folder_name = f"{picId}"
        new_folder_path = os.path.join(Data_Path, new_folder_name)

        if os.path.exists(tmpImgPath):
            os.rename(tmpImgPath, new_folder_path)
        else:
            raise FileNotFoundError(f"The folder {tmpImgPath} does not exist.")

        data_file_path = os.path.join(new_folder_path, f"{picId}.data")
        with open(data_file_path, 'w', encoding='utf-8') as data_file:
            data_file.write(f"{picId}\n")
            data_file.write(f"{picName}\n")

        self.updateData()


class Camera:
    def __init__(self, Cap_index=0):
        # 初始化摄像头
        self.cap = cv2.VideoCapture(Cap_index)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")

        # 使用 Haar 级联分类器进行人脸检测
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def readFrame(self):
        # 从摄像头读取一帧图像
        ret, frame = self.cap.read()
        if ret:
            # 将图像从 BGR 转换为 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        else:
            raise Exception("Could not read frame from camera")

    def grayImg(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return gray

    def detectFaces(self, gray, frame):
        # 使用 Haar 级联分类器检测人脸
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(150, 150)
        )
        return faces

    def getFaces(self, frame, faces, index):
        status = 0
        for (x, y, w, h) in faces:
            # 保存图像，把RGB图片看成二维数组来检测人脸区域，这里是保存在data缓冲文件夹内
            cv2.imwrite(f"./data/_tmp_img_/User.{index}.jpg", frame[y:y + h, x:x + w])
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
        data_manager.WriteTmpPicDir(self.picName, self.picId)
        self.accept()

        # 在关闭 EntWindow 后弹出 TrainWindow
        train_window = TrainWindow(self.picName, self.picId)
        train_window.exec_()


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

class TrainWorker(QThread):
    training_finished = pyqtSignal(bool)

    def __init__(self, folder, model_file_path):
        super().__init__()
        self.folder = folder.encode('utf-8').decode('utf-8')
        self.model_file_path = model_file_path.encode('utf-8').decode('utf-8')

    def run(self):
        try:
            faces, labels = self.prepareData(self.folder)
            face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            face_recognizer.train(faces, np.array(labels))
            face_recognizer.save(self.model_file_path)
            time.sleep(2)
            self.training_finished.emit(True)
        except Exception as e:
            print(f"Error in training: {e}")
            time.sleep(2)
            self.training_finished.emit(False)

    def prepareData(self, folder_path):
        faces = []
        labels = []

        # 获取所有图片文件
        image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        for image_file in image_files:
            if image_file.endswith(".jpg"):
                image_path = os.path.join(folder_path, image_file)
                image_path = image_path.encode('utf-8').decode('utf-8')

                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # 检查图像是否成功加载
                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue

                try:
                    label = int(image_file.split('.')[1])
                except ValueError as e:
                    print(f"Failed to extract label from filename: {image_file}")
                    continue

                faces.append(image)
                labels.append(label)

        return faces, labels
class TrainWindow(QDialog, Window.Ui_TrainWindow):
    def __init__(self, picName, picId, parent=None):
        super(TrainWindow, self).__init__(parent)

        # 设置无标题栏窗口
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setWindowModality(Qt.ApplicationModal)
        self.setWindowFlag(Qt.WindowStaysOnTopHint)

        self.setupUi(self)

        # 传入的 picName 和 picID
        self.picName = picName
        self.picId = picId

        # 开始训练
        self.startTraining()

    def startTraining(self):
        folder = os.path.join(Data_Path, f"{self.picId}")
        folder = str(folder)  # 确保路径是字符串
        if os.path.exists(folder):
            model_file_name = f"{self.picId}.yml"
            model_file_path = os.path.join(folder, model_file_name)
            model_file_path = str(model_file_path)  # 确保路径是字符串

            self.train_worker = TrainWorker(folder, model_file_path)
            self.train_worker.training_finished.connect(self.finishTraining)
            self.train_worker.start()

    def finishTraining(self, success):
        if success:
            print("Training completed successfully")
            self.accept()
        else:
            print("Training failed")
            self.reject()



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
            frame = camera.readFrame()
            gray = camera.grayImg(frame)
            faces = camera.detectFaces(gray, frame)

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

        data_manager.CreateTmpImgDir()

        try:
            index = 0
            while(index < Img_Num) :
                frame = camera.readFrame()
                gray = camera.grayImg(frame)
                faces = camera.detectFaces(gray, frame)

                # 在图像中绘制略大于人脸的矩形框（蓝色边框）
                for (x, y, w, h) in faces:
                    padding = 20  # 调整这个值以改变边框的大小
                    x1 = max(x - padding, 0)
                    y1 = max(y - padding, 0)
                    x2 = min(x + w + padding, frame.shape[1])
                    y2 = min(y + h + padding, frame.shape[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # 保存检测到的人脸
                status = camera.getFaces(frame, faces, index)
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
                self.ui.loadingBar.setValue(int((index / Img_Num) * 100))
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

    # 创建 DataManager 的实例，自动调用 __init__ 方法
    data_manager = DataManager()
    camera = Camera()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

