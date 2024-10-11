# -*- coding: utf-8 -*-

import os
import sys
import time
from typing import Union

import cv2
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread, QRegExp
from PyQt5.QtGui import QPixmap, QImage, QRegExpValidator
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QMessageBox, QTableWidgetItem

import Window
import DataManager

# 全局实例
camera = None  # 全局摄像头实例
data_manager = None  # 全局数据管理器实例
msgbox = None # 全局消息框实例

# 类定义
class MessageBox:
    @staticmethod
    def warning(parent = None, message = None):
        QMessageBox.warning(parent, "警告", message)

    @staticmethod
    def info(parent = None, message = None):
        QMessageBox.information(parent, "信息", message)

    @staticmethod
    def error(parent = None, message = None):
        QMessageBox.critical(parent, "错误", message)

class Camera:
    def __init__(self):
        # 初始化摄像头
        self.cam = cv2.VideoCapture(Cam_Id)  # 打开摄像头设备，默认索引为 0
        if not self.cam.isOpened():  # 检查摄像头是否成功打开
            print("摄像头初始化失败，程序中断！")
            sys.exit()

    def read_frame(self):
        # 从摄像头读取一帧图像
        ret, frame = self.cam.read()  # 读取摄像头的一帧图像，ret 表示是否成功，frame 为读到的图像
        if ret:  # 如果成功读取到图像
            # 将图像从 BGR 转换为 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame  # 返回转换后的图像

    @staticmethod
    def detect_faces(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # 使用 Haar 级联分类器检测人脸
        faces = data_manager.face_cascade.detectMultiScale(
            gray,               # 输入的灰度图像
            scaleFactor=1.1,    # 每次图像尺寸减小的比例因子
            minNeighbors=5,     # 每个候选矩形应保留的最低邻居数
            minSize=(150, 150)  # 检测窗口的最小尺寸
        )
        return faces  # 返回检测到的人脸列表

    def detect_and_draw_faces(self, frame):
        faces = self.detect_faces(frame)
        for (x, y, w, h) in faces:
            padding = 20
            x1 = max(x - padding, 0)
            y1 = max(y - padding, 0)
            x2 = min(x + w + padding, frame.shape[1])
            y2 = min(y + h + padding, frame.shape[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return faces, frame

    @staticmethod
    def save_img(frame, faces, index):
        status = 0  # 初始化状态变量为0
        for (x, y, w, h) in faces:  # 遍历检测到的人脸
            # 保存图像，把RGB图片看成二维数组来检测人脸区域，这里是保存在data缓冲文件夹内
            # 将人脸区域保存为图像文件
            cv2.imwrite(
                f"./data/_tmp_img_/User.{index}.jpg", frame[y:y + h, x:x + w])
            status = 1  # 如果成功保存图像，状态变量设置为1
        return status  # 返回状态变量

    def release(self):
        self.cam.release()

class TrainWorker(QThread):

    training_finished = pyqtSignal(bool)  # 定义一个信号，表示训练是否完成

    def __init__(self, folder, model_file_path):
        super().__init__()  # 调用父类的构造函数
        self.folder = folder.encode('utf-8').decode('utf-8')  # 处理文件夹路径，确保编码正确
        self.model_file_path = model_file_path.encode('utf-8').decode('utf-8')  # 处理模型文件路径，确保编码正确

    def run(self):
        try:
            faces, labels = self.prepare_data(self.folder)  # 准备训练数据
            recognizer = cv2.face.LBPHFaceRecognizer_create()  # 创建LBPH人脸识别器
            recognizer.train(faces, np.array(labels))  # 训练人脸识别器
            recognizer.save(self.model_file_path)  # 保存训练好的模型
            time.sleep(2)  # 暂停2秒
            self.training_finished.emit(True)  # 发送训练完成信号，表示成功完成
        except Exception as e:
            print(f"Error in training: {e}")  # 打印错误信息
            time.sleep(2)  # 暂停2秒
            self.training_finished.emit(False)  # 发送训练完成信号，表示失败

    @staticmethod
    def prepare_data(folder_path):
        faces = []  # 初始化人脸图像列表
        labels = []  # 初始化标签列表

        # 获取所有图片文件
        image_files = [f for f in os.listdir(folder_path) if os.path.isfile(
            os.path.join(folder_path, f))]  # 列出文件夹中的所有图片文件

        for image_file in image_files:
            if image_file.endswith(".jpg"):  # 检查文件是否以.jpg结尾
                image_path = os.path.join(folder_path, image_file)  # 构造图片路径
                image_path = image_path.encode('utf-8').decode('utf-8')  # 处理图片路径，确保编码正确

                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度图像形式读取图片

                # 检查图像是否成功加载
                if image is None:
                    print(f"Failed to load image: {image_path}")  # 打印加载失败的图片路径
                    continue  # 跳过当前图片

                try:
                    label = int(image_file.split('.')[1])  # 从文件名中提取标签
                except ValueError as e:
                    # 打印提取标签失败的文件名
                    print(
                        f"Failed to extract label from filename: {image_file}")
                    continue  # 跳过当前图片

                faces.append(image)  # 添加图像到人脸图像列表
                labels.append(label)  # 添加标签到标签列表

        return faces, labels  # 返回人脸图像和标签列表

class TrainWindow(QDialog, Window.Ui_TrainWindow):
    def __init__(self, img_name, img_id, parent=None):
        super(TrainWindow, self).__init__(parent)  # 调用父类的构造函数

        # 设置无标题栏窗口
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)  # 设置窗口无标题栏和对话框属性
        self.setWindowModality(Qt.ApplicationModal)  # 设置窗口为应用程序模态
        self.setWindowFlag(Qt.WindowStaysOnTopHint)  # 设置窗口始终位于其他窗口顶层

        self.setupUi(self)  # 设置UI界面

        self.img_name = img_name
        self.img_id = img_id

        # 开始训练
        self.start_training()  # 调用开始训练的方法

    def start_training(self):
        folder = os.path.join(data_manager.Data_Path, f"{self.img_id}")  # 生成用于存储图片的文件夹路径
        folder = str(folder)  # 确保路径是字符串
        if os.path.exists(folder):  # 检查文件夹是否存在
            model_file_name = f"{self.img_id}.yml"  # 生成模型文件名
            model_file_path = os.path.join(folder, model_file_name)  # 生成模型文件路径
            model_file_path = str(model_file_path)  # 确保路径是字符串

            self.train_worker = TrainWorker(folder, model_file_path)  # 创建TrainWorker线程
            self.train_worker.training_finished.connect(self.finish_training)  # 连接训练完成信号到finishTraining方法
            self.train_worker.start()  # 启动TrainWorker线程

    def finish_training(self, success):
        if success:
            print("Training completed successfully")  # 打印训练成功信息
            self.accept()  # 接受并关闭当前窗口
        else:
            print("Training failed")  # 打印训练失败信息
            msgbox.error(self, "训练失败")
            self.reject()  # 拒绝并关闭当前窗口

class EntWindow(QDialog, Window.Ui_EntWindow):
    def __init__(self, parent=None):
        super(EntWindow, self).__init__(parent)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setupUi(self)
        regex = QRegExp("[0-9]{11}")
        reg_ex_validator = QRegExpValidator(regex, self.idEdit)
        self.idEdit.setValidator(reg_ex_validator)
        self.nameEdit.setPlaceholderText("输入姓名")
        self.idEdit.setPlaceholderText("学号11位数字且不能重复录入")
        self.setWindowFlag(QtCore.Qt.Dialog)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)
        self.okButton.clicked.connect(self.check_input)

    def check_input(self):
        self.img_name = self.nameEdit.text()
        self.img_id = self.idEdit.text()
        result = data_manager.init_img_dir(self.img_name, self.img_id)
        if result == 0:
            self.close()
        elif result == 1 or result == 2:
            self.delete()
            # QMessageBox.warning(self, "警告", "名字或学号不能为空")
            msgbox.warning(self, "名字或学号不能为空")
            return
        elif result == 3:
            self.delete()
            msgbox.warning(self, "学号应为11为数字")
            return
        elif result == 4:
            self.delete()
            msgbox.warning(self, "姓名当中不应包含‘_’")
            return
        elif result == 5:
            self.delete()
            # QMessageBox.warning(self, "警告", "学号不能重复")
            msgbox.warning(self, "学号不能重复")
            return
        # 在关闭 EntWindow 后弹出 TrainWindow
        data_manager.update_data_file()

        train_window = TrainWindow(self.img_name, self.img_id, self)
        train_window.exec_()

    def delete(self):
        self.nameEdit.clear()
        self.idEdit.clear()

class RecWindow(QDialog, Window.Ui_RecWindow):
    def __init__(self, student_id, student_name, parent=None):
        super(RecWindow, self).__init__(parent)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setupUi(self)
        self.idEdit.setText(student_id)
        self.nameEdit.setText(student_name)
        self.setWindowFlag(QtCore.Qt.Dialog)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)
        self.okButton.clicked.connect(self.close)

class MngWindow(QDialog, Window.Ui_MngWindow):
    def __init__(self, parent=None):
        super(MngWindow, self).__init__(parent)
        self.showFullScreen()
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)

        self.setupUi(self)
        self.nameEdit.setReadOnly(True)
        self.idEdit.setReadOnly(True)

        self.setWindowFlag(QtCore.Qt.Dialog)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)

        self.backButton.clicked.connect(self.close)
        self.load_table()
        self.tableWidget.cellDoubleClicked.connect(self.open_folder)
        self.tableWidget.cellClicked.connect(self.display_info)
        self.delButton.clicked.connect(self.delete_selected_row)

    def open_folder(self, row):
        folder_path = self.tableWidget.item(row, 2).text()
        data_manager.open_dir(folder_path)

    def load_table(self):
        data = data_manager.load_data()
        self.tableWidget.setRowCount(len(data))
        for row, (student_name, student_id, folder_path) in enumerate(data):
            self.tableWidget.setItem(row, 0, QTableWidgetItem(student_name))
            self.tableWidget.setItem(row, 1, QTableWidgetItem(student_id))
            self.tableWidget.setItem(row, 2, QTableWidgetItem(folder_path))

    def display_info(self, row):
        student_name = self.tableWidget.item(row, 0).text()
        student_id = self.tableWidget.item(row, 1).text()
        self.nameEdit.setText(student_name)
        self.idEdit.setText(student_id)

    def delete_selected_row(self):
        current_row = self.tableWidget.currentRow()
        if current_row < 0:
            # QMessageBox.warning(self, "错误", "请选择要删除的行")
            msgbox.warning(self, "请选择要删除的行")
            return
        folder_path = self.tableWidget.item(current_row, 2).text()
        if data_manager.delete_data(folder_path):
            # QMessageBox.information(self, "成功", "文件夹已删除")
            msgbox.info(self, "文件夹已删除")
        else:
            # QMessageBox.warning(self, "错误", "文件夹不存在")
            msgbox.error(self, "文件夹不存在")
        self.load_table()

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Window.Ui_MainWindow()
        self.ui.setupUi(self)
        self.showFullScreen()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)
        self.ui.camLabel.setStyleSheet("border: 2px solid black; background-color: black;")
        self.ui.entButton.clicked.connect(self.open_EntWindow)
        self.ui.recButton.clicked.connect(self.open_RecWindow)
        self.ui.mngButton.clicked.connect(self.open_MngWindow)
        self.ui.exitButton.clicked.connect(self.close)
        self.model_recognizers = self.load_models()

    def change_ui(self, button_opt, label_text = "状态：等待操作", bar_val = 0):
        if button_opt == False:
            self.ui.entButton.setEnabled(False)
            self.ui.recButton.setEnabled(False)
            self.ui.mngButton.setEnabled(False)
            self.ui.exitButton.setEnabled(False)
        elif button_opt == True:
            self.ui.entButton.setEnabled(True)
            self.ui.recButton.setEnabled(True)
            self.ui.mngButton.setEnabled(True)
            self.ui.exitButton.setEnabled(True)

        self.ui.loadingLabel.setText(label_text)
        self.ui.loadingBar.setValue(bar_val)


    def display_frame(self, frame):
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        if not qImg.isNull():
            pixmap = QPixmap.fromImage(qImg)
            scaled_pixmap = pixmap.scaled(self.ui.camLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ui.camLabel.setPixmap(scaled_pixmap)

    def update_frame(self):
        try:
            frame = camera.read_frame()
            faces, frame = camera.detect_and_draw_faces(frame)
            self.display_frame(frame)
        except Exception as e:
            self.timer.stop()

    def capture_img(self):
        self.change_ui(False, "状态：拍照中", 0)
        data_manager.make_tmp_dir()
        try:
            index = 0
            while index < data_manager.Img_Num:
                frame = camera.read_frame()
                faces, frame = camera.detect_and_draw_faces(frame)
                status = camera.save_img(frame, faces, index)
                if status:
                    index += 1
                self.display_frame(frame)
                self.ui.loadingBar.setValue(int((index / data_manager.Img_Num) * 100))
                QApplication.processEvents()

            self.ent_window = EntWindow(self)
            self.ent_window.exec_()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.change_ui(True, "状态：等待操作", 0)
            self.timer.start(20)

    def load_models(self):
        model_recognizers = {}
        for folder in os.listdir(data_manager.Data_Path):
            folder_path = os.path.join(data_manager.Data_Path, folder)
            if os.path.isdir(folder_path):
                model_file = os.path.join(folder_path, f"{folder}.yml")
                if os.path.exists(model_file):
                    recognizer = cv2.face.LBPHFaceRecognizer_create()
                    recognizer.read(model_file)
                    model_recognizers[folder] = recognizer
        return model_recognizers

    def recognize_face(self):
        self.change_ui(False, "状态：识别中", 0)
        self.model_recognizers = self.load_models()
        recognized_count = {}
        start_time = time.time()
        try:
            while True:
                frame = camera.read_frame()
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                faces = camera.detect_faces(frame)
                for (x, y, w, h) in faces:
                    face = gray[y:y + h, x:x + w]
                    for student_id, recognizer in self.model_recognizers.items():
                        label, confidence = recognizer.predict(face)
                        if confidence < data_manager.Rec_Confidence:
                            if student_id not in recognized_count:
                                recognized_count[student_id] = 0
                            recognized_count[student_id] += 1
                            student_name = data_manager.get_name(student_id)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        else:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                self.display_frame(frame)
                for student_id, count in recognized_count.items():
                    progress = min(int((count / data_manager.Rec_Num) * 100), 100)
                    self.ui.loadingBar.setValue(progress)
                    if count >= data_manager.Rec_Num:
                        return student_id
                current_time = time.time()
                elapsed_time = current_time - start_time
                rounded_elapsed_time = str(round(elapsed_time))
                self.ui.loadingLabel.setText(f"状态：识别中（{rounded_elapsed_time}s)")
                if elapsed_time > Time_Limit:
                    return "无识别结果"
                QApplication.processEvents()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.change_ui(True, "状态：等待操作", 0)
            self.timer.start(20)

    def open_RecWindow(self):
        data_manager.update_data_file()
        if data_manager.Data_Num == 0:
            msgbox.warning(self, "暂无人脸数据")
            return

        self.timer.stop()
        self.student_id = self.recognize_face()
        self.student_name = data_manager.get_name(self.student_id)
        self.rec_window = RecWindow(self.student_id, self.student_name, self)
        self.rec_window.exec_()

    def open_EntWindow(self):
        self.timer.stop()
        self.capture_img()

    def open_MngWindow(self):
        self.mng_window = MngWindow(self)
        self.mng_window.exec_()

    def closeEvent(self, event):
        camera.release()
        super(MainWindow, self).closeEvent(event)

def init_vars():
    global face_cascade_path, face_cascade  # 全局分类器地址、分类器
    face_cascade_path = data_manager.face_cascade_path
    face_cascade = data_manager.face_cascade

    global Cam_Id, Img_Num, Rec_Num, Rec_Confidence, Time_Limit # 全局变量
    Cam_Id = data_manager.Cam_Id
    Img_Num = data_manager.Img_Num
    Rec_Num = data_manager.Rec_Num
    Rec_Confidence = data_manager.Rec_Confidence
    Time_Limit = data_manager.Time_Limit


    global Path, Data_Path # 全局地址
    Path = data_manager.Path
    Data_Path = data_manager.Data_Path

if __name__ == "__main__":

    # 初始化全局实例
    msgbox = MessageBox()
    data_manager = DataManager.DataManager()
    init_vars()
    camera = Camera()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())