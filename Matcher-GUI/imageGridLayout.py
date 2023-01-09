import os.path as osp
import glob
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, \
    QVBoxLayout, QLabel, QHBoxLayout, QGridLayout
from PyQt5.QtGui import QPixmap, QFont, QColor
from PyQt5.QtCore import Qt
import sys
from config import config



class ImageGrid(QWidget):
    def __init__(self, imagePaths):
        super().__init__()
        self.mainLayout = QGridLayout()
        self.mainLayout.addWidget(QLabel("Redwan"), 0, 0)
        self.mainLayout.addWidget(QLabel('green'), 1, 0)
        self.mainLayout.addWidget(QLabel('blue'), 1, 1)
        self.mainLayout.addWidget(QLabel('purple'), 2, 1)
        self.setLayout(self.mainLayout)
        self.mainLayout.addWidget(QLabel("This is a nice layout"))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageGrid()
    window.show()
    sys.exit(app.exec_())



