import os.path as osp
import glob
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import sys

class SingleID(QWidget):
    def __init__(self, imageFolder, ID):
        super().__init__()
        mainLayout = QVBoxLayout()
        # Making the Image Display layout
        imgFiles = []
        imgFormats = ["jpg", "png", "jpeg", "bmp"]
        for imgFormat in imgFormats:
            imgFiles.extend(glob.glob(osp.join(imageFolder, f"{ID}_*.{imgFormat}")))
            imgFiles.extend(glob.glob(osp.join(imageFolder, f"{ID}_*.{imgFormat.upper()}")))

        imageLayout = QHBoxLayout()
        for imgFile in imgFiles:
            imageLayout.addWidget(self.getImageWidget(imageURL=imgFile))


        mainLayout.addLayout(imageLayout)

        self.setLayout(mainLayout)
        # self.setMaximumHeight(1024)
        # self.setMaximumWidth(1024)


    @staticmethod
    def getImageWidget(imageURL):
        """
        This method takes the URL of an image and then makes an 
        QtWidget with image and image name at the bottom
        """
        layout = QVBoxLayout()
        imageLabel = QLabel()
        widget = QLabel()
        imageLabel.setPixmap(QPixmap(imageURL).scaled(512, 512, Qt.KeepAspectRatio))
        imageLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(imageLabel)
        textLabel = QLabel(imageURL.split("/")[-1])
        textLabel.setAlignment(Qt.AlignCenter)
        textLabel.setStyleSheet("border: 1px solid black;")
        layout.addWidget(textLabel)
        widget.setLayout(layout)
        return widget

    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SingleID(imageFolder="../Annotation/images_annotation_1", ID="012")
    window.show()
    sys.exit(app.exec_())