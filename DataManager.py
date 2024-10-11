# -*- coding: utf-8 -*-

import configparser
import os
import shutil
import sys
import subprocess
from PyQt5.QtWidgets import QMessageBox
import cv2

Path = os.getcwd()
Data_Path = os.path.join(Path, "data")
class DataManager:
    def __init__(self):
        self.read_config()
        if not os.path.exists(Data_Path):
            os.makedirs(Data_Path)
        self.update_data_file()

    def read_config(self):
        config = configparser.ConfigParser()

        try:
            # 使用 open 方法指定编码
            with open('.config', 'r', encoding='utf-8') as configfile:
                config.read_file(configfile)

            # 读取人脸分类器路径
            face_cascade_path = config.get('FaceCascade', 'face_cascade_path',
                                           fallback=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.face_cascade_path = face_cascade_path
            self.face_cascade = cv2.CascadeClassifier(eval(face_cascade_path))

            # 读取系统参数
            self.Cam_Id = config.getint('SystemParameters', 'Cam_id', fallback=0)
            self.Img_Num = config.getint('SystemParameters', 'Number_of_img', fallback=5)
            self.Rec_Num = config.getint('SystemParameters', 'Number_of_recognition', fallback=20)
            self.Rec_Confidence = config.getint('SystemParameters', 'Confidence_of_recognition', fallback=50)
            self.Time_Limit = config.getint('SystemParameters', 'Time_limit', fallback=20)

        except Exception as e:
            print(f"读取配置文件出错: {e}")
            raise

        # 初始化其他参数

        self.Data_Num = 0
        self.Path = Path
        self.Data_Path = Data_Path


    def update_data_file(self):
        subfolders = [f.name for f in os.scandir(Data_Path) if f.is_dir()]
        num_subfolders = len(subfolders)
        data_file_path = os.path.join(Data_Path, '.data')
        with open(data_file_path, 'w', encoding='utf-8') as f:
            f.write(f"{num_subfolders}\n")
            for subfolder in subfolders:
                subfolder_data_file_path = os.path.join(Data_Path, subfolder, f"{subfolder}.data")
                if os.path.exists(subfolder_data_file_path):
                    with open(subfolder_data_file_path, 'r', encoding='utf-8') as subfolder_data_file:
                        lines = subfolder_data_file.readlines()
                        if len(lines) >= 2:
                            combined_line = f"{lines[0].strip()}_{lines[1].strip()}"
                            f.write(f"{combined_line}\n")
        with open(data_file_path, 'r', encoding='utf-8') as f:
            self.Data_Num = int(f.readline().strip())

    @staticmethod
    def load_data():
        data_file_path = os.path.join(Data_Path, '.data')
        with open(data_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]
        data = [(line.split('_')[1], line.split('_')[0], os.path.join(Data_Path, line.split('_')[0])) for line in lines]
        return data

    @staticmethod
    def open_dir(folder_path):
        try:
            if sys.platform == 'win32':
                os.startfile(folder_path)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', folder_path])
            else:
                subprocess.Popen(['xdg-open', folder_path])
        except Exception as e:
            QMessageBox.warning(None, "错误", f"无法打开文件夹: {e}")

    @staticmethod
    def make_tmp_dir():
        tmp_img_path = os.path.join(Data_Path, "_tmp_img_")
        if not os.path.exists(tmp_img_path):
            os.mkdir(tmp_img_path)
        else:
            shutil.rmtree(tmp_img_path)
            os.mkdir(tmp_img_path)

    def init_img_dir(self, img_name, img_id) -> int:
        if img_name == "":
            return 1
        if img_id == "":
            return 2
        if img_id.isdigit() == False or len(img_id) != 11:
            return 3
        if "_" in img_name:
            return 4
        tmp_dir = os.path.join(Data_Path, "_tmp_img_")
        with open(os.path.join(Data_Path, '.data'), 'r', encoding='utf-8') as file:
            lines = file.readlines()
        numbers_list = [line.split('_', 1)[0] for line in lines[1:] if '_' in line]
        if img_id in numbers_list:
            return 5
        new_folder_path = os.path.join(Data_Path, img_id)
        if os.path.exists(tmp_dir):
            os.rename(tmp_dir, new_folder_path)
        else:
            raise FileNotFoundError(f"The folder {tmp_dir} does not exist.")
        data_file_path = os.path.join(new_folder_path, f"{img_id}.data")
        with open(data_file_path, 'w', encoding='utf-8') as data_file:
            data_file.write(f"{img_id}\n")
            data_file.write(f"{img_name}\n")
        self.update_data_file()
        return 0

    def delete_data(self, folder_path):
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            self.update_data_file()
            return True
        return False

    @staticmethod
    def get_name(student_id):
        student_data_file = os.path.join(Data_Path, student_id, f"{student_id}.data")
        if os.path.exists(student_data_file):
            with open(student_data_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    return lines[1].strip()
        return "Unknown"