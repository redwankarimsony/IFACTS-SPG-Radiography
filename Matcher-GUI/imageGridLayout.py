import os.path as osp
import glob
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, \
    QVBoxLayout, QLabel, QHBoxLayout, QGridLayout
from PyQt5.QtGui import QPixmap, QFont, QColor
from PyQt5.QtCore import Qt
import sys
from config import config
from imageDisplayWidget import SingleID


class ImageGrid(QWidget):
    def __init__(self, imagePaths):
        super().__init__()
        self.imagePaths = imagePaths
        self.mainLayout = QGridLayout()
        self.setLayout(self.mainLayout)

        self.num_cols = 3

        self.initGUI()

    def initGUI(self):
        idx = 0

        if len(self.imagePaths):
            for row in range((len(self.imagePaths)//self.num_cols)+1):
                for column in range(self.num_cols):
                    widget = SingleID.getImageWidget(self.imagePaths[idx])
                    print(row, column)
                    self.mainLayout.addWidget(widget, row, column)
                    idx += 1
                    if idx >= len(self.imagePaths):
                        # break the inner loop in columns
                        break
                if idx >= len(self.imagePaths):
                    # break the outer loop for row stop
                    break


if __name__ == '__main__':
    image_paths = glob.glob("/home/sonymd/Desktop/IFACTS/IFACTS-SPG-Radiography/Annotation/images_annotation_1/*.jpg")
    print(image_paths)
    app = QApplication(sys.argv)
    window = ImageGrid(imagePaths=image_paths[:5])
    window.setFixedSize(1920, 1080)
    window.show()
    sys.exit(app.exec_())
