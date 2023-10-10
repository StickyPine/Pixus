# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window.ui'
##
## Created by: Qt User Interface Compiler version 6.5.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QMainWindow, QMenuBar, QPushButton,
    QSizePolicy, QStatusBar, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(723, 511)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.BtStart = QPushButton(self.centralwidget)
        self.BtStart.setObjectName(u"BtStart")
        self.BtStart.setGeometry(QRect(20, 50, 101, 24))
        self.BtDebug = QPushButton(self.centralwidget)
        self.BtDebug.setObjectName(u"BtDebug")
        self.BtDebug.setGeometry(QRect(20, 80, 101, 24))
        self.BtResize = QPushButton(self.centralwidget)
        self.BtResize.setObjectName(u"BtResize")
        self.BtResize.setGeometry(QRect(20, 110, 101, 24))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 723, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.BtStart.setText(QCoreApplication.translate("MainWindow", u"Start", None))
        self.BtDebug.setText(QCoreApplication.translate("MainWindow", u"Debug", None))
        self.BtResize.setText(QCoreApplication.translate("MainWindow", u"Resize window", None))
    # retranslateUi

