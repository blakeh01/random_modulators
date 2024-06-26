# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'demodulator.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class UI_Demodulator(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(840, 880)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(30, 10, 331, 51))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(10, 0, 10, 0)
        self.gridLayout.setSpacing(10)
        self.gridLayout.setObjectName("gridLayout")
        self.lbl_info_symbolrate = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.lbl_info_symbolrate.setFont(font)
        self.lbl_info_symbolrate.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_info_symbolrate.setObjectName("lbl_info_symbolrate")
        self.gridLayout.addWidget(self.lbl_info_symbolrate, 0, 1, 1, 1)
        self.cmb_demod_scheme = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.cmb_demod_scheme.setObjectName("cmb_demod_scheme")
        self.cmb_demod_scheme.addItem("")
        self.gridLayout.addWidget(self.cmb_demod_scheme, 1, 0, 1, 1)
        self.txt_symbolrate = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.txt_symbolrate.setObjectName("txt_symbolrate")
        self.gridLayout.addWidget(self.txt_symbolrate, 1, 1, 1, 1)
        self.lbl_info_demodscheme = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.lbl_info_demodscheme.setFont(font)
        self.lbl_info_demodscheme.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_info_demodscheme.setObjectName("lbl_info_demodscheme")
        self.gridLayout.addWidget(self.lbl_info_demodscheme, 0, 0, 1, 1)
        self.lbl_info_effective_bitrate = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.lbl_info_effective_bitrate.setFont(font)
        self.lbl_info_effective_bitrate.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_info_effective_bitrate.setObjectName("lbl_info_effective_bitrate")
        self.gridLayout.addWidget(self.lbl_info_effective_bitrate, 0, 2, 1, 1)
        self.lbl_effective_bitrate = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.lbl_effective_bitrate.setFont(font)
        self.lbl_effective_bitrate.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_effective_bitrate.setObjectName("lbl_effective_bitrate")
        self.gridLayout.addWidget(self.lbl_effective_bitrate, 1, 2, 1, 1)
        self.list_receivedmessages = QtWidgets.QListWidget(self.centralwidget)
        self.list_receivedmessages.setGeometry(QtCore.QRect(360, 40, 461, 261))
        self.list_receivedmessages.setObjectName("list_receivedmessages")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(30, 70, 331, 71))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.lbl_info_recpackets = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.lbl_info_recpackets.setObjectName("lbl_info_recpackets")
        self.gridLayout_2.addWidget(self.lbl_info_recpackets, 0, 0, 1, 1)
        self.lbl_info_dppackets = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.lbl_info_dppackets.setObjectName("lbl_info_dppackets")
        self.gridLayout_2.addWidget(self.lbl_info_dppackets, 1, 0, 1, 1)
        self.lbl_receivedpackets = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.lbl_receivedpackets.setObjectName("lbl_receivedpackets")
        self.gridLayout_2.addWidget(self.lbl_receivedpackets, 0, 1, 1, 1)
        self.lbl_droppedpackets = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.lbl_droppedpackets.setObjectName("lbl_droppedpackets")
        self.gridLayout_2.addWidget(self.lbl_droppedpackets, 1, 1, 1, 1)
        self.btn_start = QtWidgets.QPushButton(self.centralwidget)
        self.btn_start.setGeometry(QtCore.QRect(70, 272, 261, 31))
        self.btn_start.setObjectName("btn_start")
        self.lbl_indicator = QtWidgets.QLabel(self.centralwidget)
        self.lbl_indicator.setGeometry(QtCore.QRect(36, 270, 31, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.lbl_indicator.setFont(font)
        self.lbl_indicator.setStyleSheet("color: red")
        self.lbl_indicator.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_indicator.setObjectName("lbl_indicator")
        self.lbl_info_receivedmessages = QtWidgets.QLabel(self.centralwidget)
        self.lbl_info_receivedmessages.setGeometry(QtCore.QRect(360, 10, 461, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.lbl_info_receivedmessages.setFont(font)
        self.lbl_info_receivedmessages.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_info_receivedmessages.setObjectName("lbl_info_receivedmessages")
        self.constellation_widget = PlotWidget(self.centralwidget)
        self.constellation_widget.setGeometry(QtCore.QRect(170, 320, 500, 500))
        self.constellation_widget.setObjectName("constellation_widget")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 840, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuAbout = QtWidgets.QMenu(self.menubar)
        self.menuAbout.setObjectName("menuAbout")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuAbout.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Demodulator GUI"))
        self.lbl_info_symbolrate.setText(_translate("MainWindow", "Symbol Rate"))
        self.cmb_demod_scheme.setItemText(0, _translate("MainWindow", "16-QAM"))
        self.lbl_info_demodscheme.setText(_translate("MainWindow", "Demodulation Scheme"))
        self.lbl_info_effective_bitrate.setText(_translate("MainWindow", "Effective Bitrate"))
        self.lbl_effective_bitrate.setText(_translate("MainWindow", "4800 bps"))
        self.lbl_info_recpackets.setText(_translate("MainWindow", "Received Packets:"))
        self.lbl_info_dppackets.setText(_translate("MainWindow", "Dropped Packets:"))
        self.lbl_receivedpackets.setText(_translate("MainWindow", "0"))
        self.lbl_droppedpackets.setText(_translate("MainWindow", "0"))
        self.btn_start.setText(_translate("MainWindow", "Begin Demodulating"))
        self.lbl_indicator.setText(_translate("MainWindow", "•"))
        self.lbl_info_receivedmessages.setText(_translate("MainWindow", "Received Messages"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuAbout.setTitle(_translate("MainWindow", "About"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
from pyqtgraph import PlotWidget
