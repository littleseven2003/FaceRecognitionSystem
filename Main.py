# -*- coding: utf-8 -*-
import os
import shutil
import sys
import time
import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread, QRegExp
from PyQt5.QtGui import QPixmap, QImage, QRegExpValidator,QFont
import Window
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QMessageBox,QLabel,QVBoxLayout


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
        global Data_Num  # 使用全局变量 Data_Num

        # 统计 data 文件夹下的子文件夹
        subfolders = [f.name for f in os.scandir(
            Data_Path) if f.is_dir()]  # 获取所有子文件夹的名称
        num_subfolders = len(subfolders)  # 计算子文件夹数量

        # 创建或覆盖 .data 文件
        data_file_path = os.path.join(Data_Path, '.data')  # 定义 .data 文件的路径
        with open(data_file_path, 'w', encoding='utf-8') as f:  # 打开 .data 文件，准备写入
            f.write(f"{num_subfolders}\n")  # 写入子文件夹数量
            for subfolder in subfolders:  # 遍历每个子文件夹
                subfolder_data_file_path = os.path.join(
                    Data_Path, subfolder, f"{subfolder}.data")  # 定义子文件夹内 .data 文件路径
                if os.path.exists(subfolder_data_file_path):  # 检查子文件夹内的 .data 文件是否存在
                    with open(subfolder_data_file_path, 'r',
                              encoding='utf-8') as subfolder_data_file:  # 打开子文件夹内的 .data 文件
                        lines = subfolder_data_file.readlines()  # 读取所有行
                        if len(lines) >= 2:  # 确保至少有两行数据
                            # 读取前两行并用 "_" 连接
                            # 合并前两行并去除空白字符
                            combined_line = f"{lines[0].strip()}_{lines[1].strip()}"
                            f.write(f"{combined_line}\n")  # 写入合并后的行

        # 读取 .data 文件的第一行并将其写入全局变量
        with open(data_file_path, 'r', encoding='utf-8') as f:  # 打开 .data 文件，准备读取s
            # 读取第一行并转换为整数，赋值给全局变量 Data_Num
            Data_Num = int(f.readline().strip())

    def CreateTmpImgDir(self):
        tmpImgPath = os.path.join(Data_Path, "_tmp_img_")  # 定义临时图片文件夹的路径
        if not os.path.exists(tmpImgPath):  # 检查临时图片文件夹是否不存在
            os.mkdir(tmpImgPath)  # 如果不存在，则创建该文件夹
        else:  # 如果文件夹已经存在
            shutil.rmtree(tmpImgPath)  # 删除现有的临时图片文件夹以及其内容
            os.mkdir(tmpImgPath)  # 重新创建一个新的临时图片文件夹

    def WriteTmpPicDir(self, picName, picId) -> int:
        if picName == "":
            return 1  # 名字为空的状态码
        if picId == "":
            return 2  # 学号为空的状态码
        tmpImgPath = os.path.join(Data_Path, "_tmp_img_")  # 定义临时图片文件夹的路径
        # 打开并读取 .data 文件
        with open('data/.data', 'r', encoding='utf-8') as file:
            lines = file.readlines()
        # 第一行是总行数，跳过它
        total_lines = int(lines[0].strip())  # 获取第一行的行数
        # 创建一个空的字符串数组用于存储数字
        numbers_list = []
        # 从第二行开始处理
        for line in lines[1:]:
            # 去除换行符并按照下划线分隔
            line = line.strip()
            if '_' in line:
                number, name = line.split('_', 1)  # 1 表示只分割一次，防止名字里有下划线
                numbers_list.append(number)  # 将数字添加到字符串数组中
            else:
                print(f"Invalid format: {line}")
        for stunu in numbers_list:
            if picId == stunu:
                # raise Exception("The picId already exists.")
                return 3  # 学号为重复的状态码
        new_folder_name = f"{picId}"  # 生成新的文件夹名称，以 picId 命名
        new_folder_path = os.path.join(Data_Path, new_folder_name)  # 定义新文件夹的路径

        if os.path.exists(tmpImgPath):  # 检查临时图片文件夹是否存在
            os.rename(tmpImgPath, new_folder_path)  # 将临时图片文件夹重命名为新文件夹
        else:  # 如果临时图片文件夹不存在
            raise FileNotFoundError(
                f"The folder {tmpImgPath} does not exist.")  # 抛出文件不存在的异常

        data_file_path = os.path.join(
            new_folder_path, f"{picId}.data")  # 定义新文件夹内 .data 文件的路径
        with open(data_file_path, 'w', encoding='utf-8') as data_file:  # 打开 .data 文件，准备写入
            data_file.write(f"{picId}\n")  # 写入 picId
            data_file.write(f"{picName}\n")  # 写入 picName

        self.updateData()  # 调用 updateData 方法更新数据
        return 0  # 返回0表示成功


class Camera:
    def __init__(self, Cap_index=0):
        # 初始化摄像头
        self.cap = cv2.VideoCapture(Cap_index)  # 打开摄像头设备，默认索引为 0
        if not self.cap.isOpened():  # 检查摄像头是否成功打开
            # raise Exception("Could not open video device")  # 如果摄像头未成功打开，抛出异常
            QMessageBox.warning(self, "警告", "摄像头未成功打开")  # 显示警告消息框
        # 使用 Haar 级联分类器进行人脸检测
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # 加载预训练的 Haar 级联分类器，用于检测人脸

    def readFrame(self):
        # 从摄像头读取一帧图像
        ret, frame = self.cap.read()  # 读取摄像头的一帧图像，ret 表示是否成功，frame 为读到的图像
        if ret:  # 如果成功读取到图像
            # 将图像从 BGR 转换为 RGB
            # 将图像的颜色空间从 BGR 转换为 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame  # 返回转换后的图像
        # else:  # 如果未能成功读取图像
            # raise Exception("Could not read frame from camera")  # 抛出异常

    def grayImg(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # 将图像从 RGB 转换为灰度图像
        return gray  # 返回转换后的灰度图像

    def detectFaces(self, gray, frame):
        # 使用 Haar 级联分类器检测人脸
        faces = self.face_cascade.detectMultiScale(
            gray,  # 输入的灰度图像
            scaleFactor=1.1,  # 每次图像尺寸减小的比例因子
            minNeighbors=5,  # 每个候选矩形应保留的最低邻居数
            minSize=(150, 150)  # 检测窗口的最小尺寸
        )
        return faces  # 返回检测到的人脸列表

    def getFaces(self, frame, faces, index):
        status = 0  # 初始化状态变量为0
        for (x, y, w, h) in faces:  # 遍历检测到的人脸
            # 保存图像，把RGB图片看成二维数组来检测人脸区域，这里是保存在data缓冲文件夹内
            # 将人脸区域保存为图像文件
            cv2.imwrite(
                f"./data/_tmp_img_/User.{index}.jpg", frame[y:y + h, x:x + w])
            status = 1  # 如果成功保存图像，状态变量设置为1
        return status  # 返回状态变量

    def release(self):
        # 释放摄像头资源
        self.cap.release()

# class EmptyWindow(QDialog):
#     def __init__(self, parent=None):
#         super(EmptyWindow, self).__init__(parent)  # 调用父类的构造函数
#         # 设置窗口无边框
#         self.setWindowFlags(Qt.FramelessWindowHint)
#         # 锁只能交互该窗口
#         self.setWindowModality(QtCore.Qt.ApplicationModal)
#         # 始终顶端
#         self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)
#         self.setWindowOpacity(0.0)
#         # layout = QVBoxLayout(self)
#         # self.helloLabel = QLabel("拍照中", self)
#         # self.layout().setContentsMargins(50,50,50,50)
#         # # 设置字体大小
#         # font = QFont()
#         # font.setPointSize(70)
#         # self.helloLabel.setFont(font)
#         # # 设置标签的对齐方式，以确保文本在标签中居中
#         # self.helloLabel.setAlignment(Qt.AlignCenter)
#         # layout.addWidget(self.helloLabel)
#     def closeEvent(self, event):
#     # 阻止窗口关闭
#         event.ignore()

class EntWindow(QDialog, Window.Ui_EntWindow):
    def __init__(self, parent=None):
        super(EntWindow, self).__init__(parent)

        # 获取当前的窗口标志
        # current_flags = self.windowFlags()
        # 设置禁用最大化、最小化和关闭按钮
        # self.setWindowFlags(
        #    current_flags & ~QtCore.Qt.WindowMinimizeButtonHint & ~QtCore.Qt.WindowMaximizeButtonHint & ~QtCore.Qt.WindowCloseButtonHint)

        # 设置无标题栏窗口
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)

        self.setupUi(self)
        regex = QRegExp("[0-9]{11}")  # 正则表达式：只允许输入11位数字
        reg_ex_validator = QRegExpValidator(regex, self.idEdit)
        self.idEdit.setValidator(reg_ex_validator)
        self.nameEdit.setPlaceholderText("输入姓名")
        self.idEdit.setPlaceholderText("学号11位数字且不能重复录入")
        self.setWindowFlag(QtCore.Qt.Dialog)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)

        self.okButton.clicked.connect(self.inputName)

    def inputName(self):

        self.picName = self.nameEdit.text()  # 从文本编辑框获取用户输入的姓名
        self.picId = self.idEdit.text()  # 从文本编辑框获取用户输入的ID
        result = data_manager.WriteTmpPicDir(
            self.picName, self.picId)  # 将姓名和ID写入临时图片目录
        if result == 0:
            self.close()  # 关闭当前窗口
        elif result == 1:
            self.delete()
            QMessageBox.warning(self, "警告", "名字或学号不能为空")  # 显示警告消息框
            return
        elif result == 2:
            self.delete()
            QMessageBox.warning(self, "警告", "名字或学号不能为空")  # 显示警告消息框
            return
        elif result == 3:
            self.delete()
            QMessageBox.warning(self, "警告", "学号不能重复")  # 显示警告消息框
            return

        # if success:
        #     self.close()  # 关闭当前窗口

        # else:
        #     self.delete()
        #     QMessageBox.warning(self, "警告", "输入信息有误，请重新输入")  # 显示警告消息框
        #     return
        # self.accept()  # 接受输入并关闭当前窗口

        # 在关闭 EntWindow 后弹出 TrainWindow
        train_window = TrainWindow(
            self.picName, self.picId, self)  # 创建TrainWindow窗口实例
        train_window.exec_()  # 以模态形式显示TrainWindow窗口

    def delete(self):
        self.nameEdit.clear()
        self.idEdit.clear()


class RecWindow(QDialog, Window.Ui_RecWindow):
    def __init__(self, parent=None):
        super(RecWindow, self).__init__(parent)

        # 获取当前的窗口标志
        # current_flags = self.windowFlags()
        # 设置禁用最大化、最小化和关闭按钮
        # self.setWindowFlags(
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
    training_finished = pyqtSignal(bool)  # 定义一个信号，表示训练是否完成

    def __init__(self, folder, model_file_path):
        super().__init__()  # 调用父类的构造函数
        self.folder = folder.encode('utf-8').decode('utf-8')  # 处理文件夹路径，确保编码正确
        self.model_file_path = model_file_path.encode(
            'utf-8').decode('utf-8')  # 处理模型文件路径，确保编码正确

    def run(self):
        try:
            faces, labels = self.prepareData(self.folder)  # 准备训练数据
            face_recognizer = cv2.face.LBPHFaceRecognizer_create()  # 创建LBPH人脸识别器
            face_recognizer.train(faces, np.array(labels))  # 训练人脸识别器
            face_recognizer.save(self.model_file_path)  # 保存训练好的模型
            time.sleep(2)  # 暂停2秒
            self.training_finished.emit(True)  # 发送训练完成信号，表示成功完成
        except Exception as e:
            print(f"Error in training: {e}")  # 打印错误信息
            time.sleep(2)  # 暂停2秒
            self.training_finished.emit(False)  # 发送训练完成信号，表示失败

    def prepareData(self, folder_path):
        faces = []  # 初始化人脸图像列表
        labels = []  # 初始化标签列表

        # 获取所有图片文件
        image_files = [f for f in os.listdir(folder_path) if os.path.isfile(
            os.path.join(folder_path, f))]  # 列出文件夹中的所有图片文件

        for image_file in image_files:
            if image_file.endswith(".jpg"):  # 检查文件是否以.jpg结尾
                image_path = os.path.join(folder_path, image_file)  # 构造图片路径
                image_path = image_path.encode(
                    'utf-8').decode('utf-8')  # 处理图片路径，确保编码正确

                image = cv2.imread(
                    image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度图像形式读取图片

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
    def __init__(self, picName, picId, parent=None):
        super(TrainWindow, self).__init__(parent)  # 调用父类的构造函数

        # 设置无标题栏窗口
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)  # 设置窗口无标题栏和对话框属性
        self.setWindowModality(Qt.ApplicationModal)  # 设置窗口为应用程序模态
        self.setWindowFlag(Qt.WindowStaysOnTopHint)  # 设置窗口始终位于其他窗口顶层

        self.setupUi(self)  # 设置UI界面

        # 传入的 picName 和 picID
        self.picName = picName  # 存储传入的picName
        self.picId = picId  # 存储传入的picId

        # 开始训练
        self.startTraining()  # 调用开始训练的方法

    def startTraining(self):
        folder = os.path.join(Data_Path, f"{self.picId}")  # 生成用于存储图片的文件夹路径
        folder = str(folder)  # 确保路径是字符串
        if os.path.exists(folder):  # 检查文件夹是否存在
            model_file_name = f"{self.picId}.yml"  # 生成模型文件名
            model_file_path = os.path.join(folder, model_file_name)  # 生成模型文件路径
            model_file_path = str(model_file_path)  # 确保路径是字符串

            self.train_worker = TrainWorker(
                folder, model_file_path)  # 创建TrainWorker线程
            self.train_worker.training_finished.connect(
                self.finishTraining)  # 连接训练完成信号到finishTraining方法
            self.train_worker.start()  # 启动TrainWorker线程

    def finishTraining(self, success):
        if success:
            print("Training completed successfully")  # 打印训练成功信息
            self.accept()  # 接受并关闭当前窗口
        else:
            print("Training failed")  # 打印训练失败信息
            self.reject()  # 拒绝并关闭当前窗口


class MngWindow(QDialog, Window.Ui_MngWindow):
    def __init__(self, parent=None):
        super(MngWindow, self).__init__(parent)
        self.showFullScreen()  # 设置窗口全屏显示

        # 获取当前的窗口标志
        # current_flags = self.windowFlags()
        # 设置禁用最大化、最小化和关闭按钮
        # self.setWindowFlags(
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
        self.ui.camLabel.setStyleSheet(
            "border: 2px solid black; background-color: black;")

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
            frame = camera.readFrame()  # 读取相机帧
            gray = camera.grayImg(frame)  # 将帧转换为灰度图像
            faces = camera.detectFaces(gray, frame)  # 检测人脸

            # 在图像中绘制略大于人脸的矩形框（红色边框）
            for (x, y, w, h) in faces:
                padding = 20  # 调整这个值以改变边框的大小
                x1 = max(x - padding, 0)  # 计算矩形框的左上角x坐标，确保不超出边界
                y1 = max(y - padding, 0)  # 计算矩形框的左上角y坐标，确保不超出边界
                # 计算矩形框的右下角x坐标，确保不超出边界
                x2 = min(x + w + padding, frame.shape[1])
                # 计算矩形框的右下角y坐标，确保不超出边界
                y2 = min(y + h + padding, frame.shape[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 0, 255), 2)  # 在帧上绘制红色矩形框

            # 将 OpenCV 图像转换为 QImage
            height, width, channel = frame.shape  # 获取帧的高度、宽度和通道数
            bytesPerLine = 3 * width  # 计算每行的字节数
            qImg = QImage(frame.data, width, height, bytesPerLine,
                          QImage.Format_RGB888)  # 创建QImage对象

            if qImg.isNull():
                print("Warning: QImage is null")  # 如果QImage为空，打印警告信息
            else:
                pixmap = QPixmap.fromImage(qImg)  # 将QImage转换为QPixmap
                scaled_pixmap = pixmap.scaled(self.ui.camLabel.size(), Qt.KeepAspectRatio,
                                              Qt.SmoothTransformation)  # 缩放QPixmap
                self.ui.camLabel.setPixmap(scaled_pixmap)  # 在标签上显示QPixmap
        except Exception as e:
            # print(f"Error: {e}")  # 打印错误信息
            # QMessageBox.warning(self, "警告", "读取摄像头失败")  # 显示警告消息框
            self.timer.stop()  # 停止定时器

    def captureImg(self):
        # 禁用所有按钮
        self.ui.entButton.setEnabled(False)
        self.ui.recButton.setEnabled(False)
        self.ui.mngButton.setEnabled(False)
        self.ui.exitButton.setEnabled(False)


        #time.sleep(5)

        # 设置状态为拍照中
        self.ui.loadingLabel.setText("状态：拍照中")  # 更新状态标签文本
        self.ui.loadingBar.setValue(0)  # 将进度条重置为0

        data_manager.CreateTmpImgDir()  # 创建临时图片目录

        try:
            index = 0  # 初始化图像索引
            while index < Img_Num:
                frame = camera.readFrame()  # 读取相机帧
                gray = camera.grayImg(frame)  # 将帧转换为灰度图像
                faces = camera.detectFaces(gray, frame)  # 检测人脸

                # 在图像中绘制略大于人脸的矩形框（红色边框）
                for (x, y, w, h) in faces:
                    padding = 20  # 调整这个值以改变边框的大小
                    x1 = max(x - padding, 0)  # 计算矩形框的左上角x坐标，确保不超出边界
                    y1 = max(y - padding, 0)  # 计算矩形框的左上角y坐标，确保不超出边界
                    x2 = min(x + w + padding, frame.shape[1])
                    y2 = min(y + h + padding, frame.shape[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # 保存检测到的人脸
                status = camera.getFaces(frame, faces, index)  # 获取人脸图像并保存
                if status:
                    index += 1  # 如果成功保存人脸图像，增加索引

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
            self.ui.loadingLabel.setText("状态：等待操作")  # 更新状态标签文本
            self.ui.loadingBar.setValue(0)  # 将进度条重置为0
            self.ent_window = EntWindow(self)  # 创建EntWindow窗口实例
            self.ent_window.exec_()  # 以模态形式显示EntWindow窗口

        except Exception as e:
            print(f"Error: {e}")  # 打印错误信息
        finally:
            # 启用所有按钮
            self.ui.entButton.setEnabled(True)
            self.ui.recButton.setEnabled(True)
            self.ui.mngButton.setEnabled(True)
            self.ui.exitButton.setEnabled(True)

            # 恢复 updateFrame 调用
            self.timer.start(20)  # 启动定时器，每20毫秒调用一次updateFrame

    def closeEvent(self, event):
        camera.release()
        super(MainWindow, self).closeEvent(event)

    def openRecWindow(self):
        self.rec_window = RecWindow(self)
        self.rec_window.exec_()

    def openEntWindow(self):
        self.timer.stop()  # 停止 updateFrame

        self.captureImg()  # 调用拍照方法
        # self.ent_window = EntWindow(self)
        # self.ent_window.exec_()

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
