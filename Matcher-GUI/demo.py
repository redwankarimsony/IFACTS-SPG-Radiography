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
        imageFiles = glob.glob(f"{imageFolder}/{ID}*.jpg")
        imageLayout = QHBoxLayout()
        for imageFile in imageFiles:
            imageLayout.addWidget(self.getImageWidget(imageURL=imageFile))


        mainLayout.addLayout(imageLayout)

        self.setLayout(mainLayout)
        self.setMaximumHeight(1024)
        self.setMaximumWidth(1024)


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