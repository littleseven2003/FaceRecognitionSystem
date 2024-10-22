
from PyQt5 import QtCore, QtGui, QtWidgets
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
        self.camLabel = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.camLabel.sizePolicy().hasHeightForWidth())
        self.camLabel.setSizePolicy(sizePolicy)
        self.camLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.camLabel.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.camLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.camLabel.setObjectName("camLabel")
        self.CamLayout_2.addWidget(self.camLabel)
        self.BarLayout = QtWidgets.QVBoxLayout()
        self.BarLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.BarLayout.setObjectName("BarLayout")
        self.loadingBar = QtWidgets.QProgressBar(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.loadingBar.sizePolicy().hasHeightForWidth())
        self.loadingBar.setSizePolicy(sizePolicy)
        self.loadingBar.setProperty("value", 0)
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
        self.camLabel.setText(_translate("MainWindow", "No Camera"))
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
        self.nameEdit = QtWidgets.QLineEdit(EntWindow)
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
        self.nameEdit.setObjectName("nameEdit")
        self.nameLayout.addWidget(self.nameEdit, 0, 1, 1, 1)
        self.idEdit = QtWidgets.QLineEdit(EntWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.idEdit.sizePolicy().hasHeightForWidth())
        self.idEdit.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.idEdit.setFont(font)
        self.idEdit.setObjectName("idEdit")
        self.nameLayout.addWidget(self.idEdit, 1, 1, 1, 1)
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
        self.entLayout.addLayout(self.nameLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
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
        self.horizontalLayout_2.addWidget(self.okButton)
        self.exitButton = QtWidgets.QPushButton(EntWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.exitButton.sizePolicy().hasHeightForWidth())
        self.exitButton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.exitButton.setFont(font)
        self.exitButton.setObjectName("exitButton")
        self.horizontalLayout_2.addWidget(self.exitButton)
        self.entLayout.addLayout(self.horizontalLayout_2)
        self.entLayout.setStretch(0, 3)
        self.entLayout.setStretch(1, 1)

        self.retranslateUi(EntWindow)
        QtCore.QMetaObject.connectSlotsByName(EntWindow)

    def retranslateUi(self, EntWindow):
        _translate = QtCore.QCoreApplication.translate
        EntWindow.setWindowTitle(_translate("EntWindow", "人脸录入 - 信息录入"))
        self.nameLabel.setText(_translate("EntWindow", "姓名："))
        self.idLabel.setText(_translate("EntWindow", "学号："))
        self.okButton.setText(_translate("EntWindow", "确定"))
        self.exitButton.setText(_translate("EntWindow", "取消"))

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

class Ui_MngWindow(object):
    def setupUi(self, MngWindow):
        MngWindow.setObjectName("MngWindow")
        MngWindow.resize(1024, 768)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MngWindow.sizePolicy().hasHeightForWidth())
        MngWindow.setSizePolicy(sizePolicy)
        MngWindow.setMinimumSize(QtCore.QSize(1024, 768))
        MngWindow.setMaximumSize(QtCore.QSize(1024, 768))
        self.mngLayout = QtWidgets.QHBoxLayout(MngWindow)
        self.mngLayout.setContentsMargins(30, 50, 30, 50)
        self.mngLayout.setSpacing(30)
        self.mngLayout.setObjectName("mngLayout")
        self.tableWidget = QtWidgets.QTableWidget(MngWindow)
        self.tableWidget.setMidLineWidth(1)
        self.tableWidget.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.tableWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tableWidget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tableWidget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tableWidget.setRowCount(10)
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setObjectName("tableWidget")
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 0, item)
        self.tableWidget.horizontalHeader().setCascadingSectionResizes(False)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(200)
        self.tableWidget.horizontalHeader().setMinimumSectionSize(150)
        self.tableWidget.verticalHeader().setDefaultSectionSize(60)
        self.tableWidget.verticalHeader().setMinimumSectionSize(60)
        self.mngLayout.addWidget(self.tableWidget)
        self.funLayout = QtWidgets.QVBoxLayout()
        self.funLayout.setContentsMargins(-1, 100, -1, 100)
        self.funLayout.setSpacing(50)
        self.funLayout.setObjectName("funLayout")
        self.nameLayout = QtWidgets.QGridLayout()
        self.nameLayout.setContentsMargins(-1, 50, -1, 50)
        self.nameLayout.setVerticalSpacing(20)
        self.nameLayout.setObjectName("nameLayout")
        self.nameLabel = QtWidgets.QLabel(MngWindow)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.nameLabel.setFont(font)
        self.nameLabel.setObjectName("nameLabel")
        self.nameLayout.addWidget(self.nameLabel, 0, 0, 1, 1)
        self.idLabel = QtWidgets.QLabel(MngWindow)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.idLabel.setFont(font)
        self.idLabel.setObjectName("idLabel")
        self.nameLayout.addWidget(self.idLabel, 1, 0, 1, 1)
        self.idEdit = QtWidgets.QLineEdit(MngWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.idEdit.sizePolicy().hasHeightForWidth())
        self.idEdit.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.idEdit.setFont(font)
        self.idEdit.setObjectName("idEdit")
        self.nameLayout.addWidget(self.idEdit, 1, 1, 1, 1)
        self.nameEdit = QtWidgets.QLineEdit(MngWindow)
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
        self.nameEdit.setObjectName("nameEdit")
        self.nameLayout.addWidget(self.nameEdit, 0, 1, 1, 1)
        self.funLayout.addLayout(self.nameLayout)
        self.delButton = QtWidgets.QPushButton(MngWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.delButton.sizePolicy().hasHeightForWidth())
        self.delButton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(20)
        self.delButton.setFont(font)
        self.delButton.setObjectName("delButton")
        self.funLayout.addWidget(self.delButton)
        self.backButton = QtWidgets.QPushButton(MngWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.backButton.sizePolicy().hasHeightForWidth())
        self.backButton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(20)
        self.backButton.setFont(font)
        self.backButton.setObjectName("backButton")
        self.funLayout.addWidget(self.backButton)
        self.funLayout.setStretch(0, 3)
        self.funLayout.setStretch(1, 1)
        self.funLayout.setStretch(2, 1)
        self.mngLayout.addLayout(self.funLayout)
        self.mngLayout.setStretch(0, 5)
        self.mngLayout.setStretch(1, 2)

        self.retranslateUi(MngWindow)
        QtCore.QMetaObject.connectSlotsByName(MngWindow)

    def retranslateUi(self, MngWindow):
        _translate = QtCore.QCoreApplication.translate
        MngWindow.setWindowTitle(_translate("MngWindow", "数据管理"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MngWindow", "姓名"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MngWindow", "学号"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MngWindow", "图片文件地址"))
        __sortingEnabled = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)
        self.tableWidget.setSortingEnabled(__sortingEnabled)
        self.nameLabel.setText(_translate("MngWindow", "姓名："))
        self.idLabel.setText(_translate("MngWindow", "学号："))
        self.delButton.setText(_translate("MngWindow", "删除"))
        self.backButton.setText(_translate("MngWindow", "返回"))

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


