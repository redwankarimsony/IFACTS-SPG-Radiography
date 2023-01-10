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
    def __init__(self, results):
        """

        Parameters
        ----------
        results: This is a list of lists. each of the elements of the list is another list.
        [[filepath1, label1, score1], [filepath2, label2, score2], ...]
        """
        super().__init__()
        self.results = results
        self.mainLayout = QGridLayout()
        self.setLayout(self.mainLayout)

        self.num_cols = 3

        self.initGUI()

    def initGUI(self):
        idx = 0

        if len(self.results):
            for row in range((len(self.results) // self.num_cols) + 1):
                for column in range(self.num_cols):
                    widget = SingleID.getImageWidget(self.results[idx][0],
                                                     label= self.results[idx][1],
                                                     score=self.results[idx][2] if len(self.results[idx]) == 3 else
                                                     self.results[idx][1])
                    print(row, column)
                    self.mainLayout.addWidget(widget, row, column)
                    idx += 1
                    if idx >= len(self.results):
                        # break the inner loop in columns
                        break
                if idx >= len(self.results):
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
