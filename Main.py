# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog
from PyQt5 import QtCore, QtGui, QtWidgets

# Form implementation generated from reading ui file 'Main.py'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.ApplicationModal)
        MainWindow.resize(1024, 768)
        MainWindow.setMinimumSize(QtCore.QSize(1024, 768))
        MainWindow.setMaximumSize(QtCore.QSize(1024, 768))
        MainWindow.setFocusPolicy(QtCore.Qt.NoFocus)
        MainWindow.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        MainWindow.setWhatsThis("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.mainLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.mainLayout.setContentsMargins(50, 100, 50, 100)
        self.mainLayout.setSpacing(50)
        self.mainLayout.setObjectName("mainLayout")
        self.CamLayout_2 = QtWidgets.QVBoxLayout()
        self.CamLayout_2.setSpacing(30)
        self.CamLayout_2.setObjectName("CamLayout_2")
        self.cameraWidget = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cameraWidget.sizePolicy().hasHeightForWidth())
        self.cameraWidget.setSizePolicy(sizePolicy)
        self.cameraWidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.cameraWidget.setStyleSheet("border: 2px solid black;")
        self.cameraWidget.setObjectName("cameraWidget")
        self.CamLayout_2.addWidget(self.cameraWidget)
        self.BarLayout = QtWidgets.QVBoxLayout()
        self.BarLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.BarLayout.setObjectName("BarLayout")
        self.loadingBar = QtWidgets.QProgressBar(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.loadingBar.sizePolicy().hasHeightForWidth())
        self.loadingBar.setSizePolicy(sizePolicy)
        self.loadingBar.setProperty("value", 24)
        self.loadingBar.setObjectName("loadingBar")
        self.BarLayout.addWidget(self.loadingBar)
        self.loadingLabel = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.loadingLabel.sizePolicy().hasHeightForWidth())
        self.loadingLabel.setSizePolicy(sizePolicy)
        self.loadingLabel.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.loadingLabel.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.loadingLabel.setFrameShadow(QtWidgets.QFrame.Plain)
        self.loadingLabel.setLineWidth(0)
        self.loadingLabel.setTextFormat(QtCore.Qt.PlainText)
        self.loadingLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.loadingLabel.setWordWrap(False)
        self.loadingLabel.setObjectName("loadingLabel")
        self.BarLayout.addWidget(self.loadingLabel, 0, QtCore.Qt.AlignVCenter)
        self.BarLayout.setStretch(1, 2)
        self.CamLayout_2.addLayout(self.BarLayout)
        self.CamLayout_2.setStretch(0, 50)
        self.CamLayout_2.setStretch(1, 1)
        self.mainLayout.addLayout(self.CamLayout_2)
        self.ButtonLayout = QtWidgets.QVBoxLayout()
        self.ButtonLayout.setContentsMargins(-1, 50, -1, 50)
        self.ButtonLayout.setSpacing(50)
        self.ButtonLayout.setObjectName("ButtonLayout")
        self.entButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.entButton.sizePolicy().hasHeightForWidth())
        self.entButton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(20)
        self.entButton.setFont(font)
        self.entButton.setObjectName("entButton")
        self.ButtonLayout.addWidget(self.entButton)
        self.recButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.recButton.sizePolicy().hasHeightForWidth())
        self.recButton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(20)
        self.recButton.setFont(font)
        self.recButton.setObjectName("recButton")
        self.ButtonLayout.addWidget(self.recButton)
        self.mngButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mngButton.sizePolicy().hasHeightForWidth())
        self.mngButton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(20)
        self.mngButton.setFont(font)
        self.mngButton.setObjectName("mngButton")
        self.ButtonLayout.addWidget(self.mngButton)
        self.exitButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.exitButton.sizePolicy().hasHeightForWidth())
        self.exitButton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(20)
        self.exitButton.setFont(font)
        self.exitButton.setObjectName("exitButton")
        self.ButtonLayout.addWidget(self.exitButton)
        self.mainLayout.addLayout(self.ButtonLayout)
        self.mainLayout.setStretch(0, 3)
        self.mainLayout.setStretch(1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.exitButton.clicked.connect(MainWindow.close) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "人脸识别系统"))
        self.loadingLabel.setText(_translate("MainWindow", "状态：等待操作"))
        self.entButton.setText(_translate("MainWindow", "人脸录入"))
        self.recButton.setText(_translate("MainWindow", "人脸识别"))
        self.mngButton.setText(_translate("MainWindow", "数据管理"))
        self.exitButton.setText(_translate("MainWindow", "退出"))

class Ui_EntWindow(object):
    def setupUi(self, EntWindow):
        EntWindow.setObjectName("EntWindow")
        EntWindow.resize(480, 339)
        self.entLayout = QtWidgets.QVBoxLayout(EntWindow)
        self.entLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.entLayout.setContentsMargins(50, 50, 50, 50)
        self.entLayout.setSpacing(30)
        self.entLayout.setObjectName("entLayout")
        self.nameLayout = QtWidgets.QGridLayout()
        self.nameLayout.setContentsMargins(-1, 20, -1, 20)
        self.nameLayout.setVerticalSpacing(20)
        self.nameLayout.setObjectName("nameLayout")
        self.nameLabel = QtWidgets.QLabel(EntWindow)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.nameLabel.setFont(font)
        self.nameLabel.setObjectName("nameLabel")
        self.nameLayout.addWidget(self.nameLabel, 0, 0, 1, 1)
        self.idLabel = QtWidgets.QLabel(EntWindow)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.idLabel.setFont(font)
        self.idLabel.setObjectName("idLabel")
        self.nameLayout.addWidget(self.idLabel, 1, 0, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(EntWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit.sizePolicy().hasHeightForWidth())
        self.lineEdit.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.lineEdit.setFont(font)
        self.lineEdit.setText("")
        self.lineEdit.setObjectName("lineEdit")
        self.nameLayout.addWidget(self.lineEdit, 0, 1, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(EntWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_2.sizePolicy().hasHeightForWidth())
        self.lineEdit_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.nameLayout.addWidget(self.lineEdit_2, 1, 1, 1, 1)
        self.entLayout.addLayout(self.nameLayout)
        self.okButton = QtWidgets.QPushButton(EntWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.okButton.sizePolicy().hasHeightForWidth())
        self.okButton.setSizePolicy(sizePolicy)
        self.okButton.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.okButton.setFont(font)
        self.okButton.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.okButton.setObjectName("okButton")
        self.entLayout.addWidget(self.okButton)
        self.entLayout.setStretch(0, 3)
        self.entLayout.setStretch(1, 1)

        self.retranslateUi(EntWindow)
        self.okButton.clicked.connect(EntWindow.close) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(EntWindow)

    def retranslateUi(self, EntWindow):
        _translate = QtCore.QCoreApplication.translate
        EntWindow.setWindowTitle(_translate("EntWindow", "人脸录入 - 信息录入"))
        self.nameLabel.setText(_translate("EntWindow", "姓名："))
        self.idLabel.setText(_translate("EntWindow", "学号："))
        self.okButton.setText(_translate("EntWindow", "确定"))


class EntWindow(QDialog, Ui_EntWindow):
    def __init__(self):
        super(EntWindow, self).__init__()
        self.setupUi(self)
        self.setWindowModality(QtCore.Qt.ApplicationModal)  # 设置窗口模态

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.showFullScreen()  # 设置窗口全屏显示

        # 连接 entButton 的点击信号到槽函数
        self.ui.entButton.clicked.connect(self.openEntWindow)

    def openEntWindow(self):
        self.ent_window = EntWindow()
        self.ent_window.exec_()  # 使用 exec_() 以确保 MainWindow 被禁用，EntWindow 始终在最上层

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
