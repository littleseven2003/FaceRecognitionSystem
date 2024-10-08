# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'RecWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_RecWindow(object):
    def setupUi(self, RecWindow):
        RecWindow.setObjectName("RecWindow")
        RecWindow.resize(480, 339)
        self.recLayout = QtWidgets.QVBoxLayout(RecWindow)
        self.recLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.recLayout.setContentsMargins(50, 20, 50, 50)
        self.recLayout.setSpacing(20)
        self.recLayout.setObjectName("recLayout")
        self.resultLabel = QtWidgets.QLabel(RecWindow)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.resultLabel.setFont(font)
        self.resultLabel.setObjectName("resultLabel")
        self.recLayout.addWidget(self.resultLabel)
        self.nameLayout = QtWidgets.QGridLayout()
        self.nameLayout.setContentsMargins(-1, 20, -1, 20)
        self.nameLayout.setVerticalSpacing(20)
        self.nameLayout.setObjectName("nameLayout")
        self.nameLabel = QtWidgets.QLabel(RecWindow)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.nameLabel.setFont(font)
        self.nameLabel.setObjectName("nameLabel")
        self.nameLayout.addWidget(self.nameLabel, 0, 0, 1, 1)
        self.idLabel = QtWidgets.QLabel(RecWindow)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.idLabel.setFont(font)
        self.idLabel.setObjectName("idLabel")
        self.nameLayout.addWidget(self.idLabel, 1, 0, 1, 1)
        self.nameEdit = QtWidgets.QLineEdit(RecWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.nameEdit.sizePolicy().hasHeightForWidth())
        self.nameEdit.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.nameEdit.setFont(font)
        self.nameEdit.setText("")
        self.nameEdit.setReadOnly(True)
        self.nameEdit.setObjectName("nameEdit")
        self.nameLayout.addWidget(self.nameEdit, 0, 1, 1, 1)
        self.idEdit = QtWidgets.QLineEdit(RecWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.idEdit.sizePolicy().hasHeightForWidth())
        self.idEdit.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.idEdit.setFont(font)
        self.idEdit.setReadOnly(True)
        self.idEdit.setObjectName("idEdit")
        self.nameLayout.addWidget(self.idEdit, 1, 1, 1, 1)
        self.recLayout.addLayout(self.nameLayout)
        self.okButton = QtWidgets.QPushButton(RecWindow)
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
        self.recLayout.addWidget(self.okButton)
        self.recLayout.setStretch(0, 1)
        self.recLayout.setStretch(1, 5)
        self.recLayout.setStretch(2, 1)

        self.retranslateUi(RecWindow)
        QtCore.QMetaObject.connectSlotsByName(RecWindow)

    def retranslateUi(self, RecWindow):
        _translate = QtCore.QCoreApplication.translate
        RecWindow.setWindowTitle(_translate("RecWindow", "人脸识别 - 识别结果"))
        self.resultLabel.setText(_translate("RecWindow", "识别结果："))
        self.nameLabel.setText(_translate("RecWindow", "姓名："))
        self.idLabel.setText(_translate("RecWindow", "学号："))
        self.okButton.setText(_translate("RecWindow", "确定"))
