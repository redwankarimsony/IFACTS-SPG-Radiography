import sys, os, glob
import cv2, math
import numpy as np
from PIL import Image, ImageQt
from PyQt5 import uic
from PyQt5.QtCore import QCoreApplication, QUrl
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *






class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()




class Demo(QWidget):
    def __init__(self, parent: typing.Optional['QWidget'] = ..., flags: typing.Union[QtCore.Qt.WindowFlags, QtCore.Qt.WindowType] = ...) -> None:
        super().__init__(parent, flags)
        self.styleSheet("font-size: 30px;")

        mainLayout = QHBoxLayout()
        self.layoutA = QVBoxLayout()
        mainLayout.addLayout(self.layoutA)

        buttonA = QPushButton("Print Values")
        buttonA.clicked.connect(self.)
        



if __name__ == '__main__':
    app = QApplication(sys.argv)
    visualizer = MainWindow()
    visualizer.show()
    sys.exit(app.exec_())
