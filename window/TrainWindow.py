# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'TrainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_TrainWindow(object):
    def setupUi(self, TrainWindow):
        TrainWindow.setObjectName("TrainWindow")
        TrainWindow.resize(335, 222)
        self.trainLayout = QtWidgets.QVBoxLayout(TrainWindow)
        self.trainLayout.setObjectName("trainLayout")
        self.trainLaber = QtWidgets.QLabel(TrainWindow)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(26)
        self.trainLaber.setFont(font)
        self.trainLaber.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.trainLaber.setAlignment(QtCore.Qt.AlignCenter)
        self.trainLaber.setObjectName("trainLaber")
        self.trainLayout.addWidget(self.trainLaber)

        self.retranslateUi(TrainWindow)
        QtCore.QMetaObject.connectSlotsByName(TrainWindow)

    def retranslateUi(self, TrainWindow):
        _translate = QtCore.QCoreApplication.translate
        TrainWindow.setWindowTitle(_translate("TrainWindow", "人脸识别 - 训练中"))
        self.trainLaber.setText(_translate("TrainWindow", "正在训练中..."))
