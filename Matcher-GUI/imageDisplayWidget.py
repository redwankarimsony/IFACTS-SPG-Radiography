import os.path as osp
import glob
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QHBoxLayout
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
import sys
from config import config

class SingleID(QWidget):
    def __init__(self, imageFolder, ID):
        """

        Parameters
        ----------
        imageFolder: takes the full path of the directory which contains the ID folders
        ID: Individual Id of the person, X-ray to be visualized.
        """
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


    @staticmethod
    def getImageWidget(imageURL, label = None, score=None):
        """
        This method takes the URL of an image and then makes an 
        QtWidget with image and image name at the bottom
        """

        layout = QVBoxLayout()
        imageLabel = QLabel()
        widget = QLabel()
        imageLabel.setPixmap(QPixmap(imageURL).scaled(config.displayImageWidth,
                                                      config.displayImageHeight,
                                                      Qt.KeepAspectRatio))
        imageLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(imageLabel)
        text = imageURL.split("/")[-1].split(".")[0] + str(f" {score:0.2f}") if score else imageURL.split("/")[-1]
        textLabel = QLabel(text)
        textLabel.setAlignment(Qt.AlignCenter)
        textLabel.setGeometry(0, 0, 800, 50)
        textLabel.setFixedHeight(50)
        if label ==0.0: color_code = "red"
        elif label ==1.0: color_code = "green"
        if label is not None:
            textLabel.setStyleSheet(f"background-color: {color_code}; border: 1px solid black; ")
        else:
            textLabel.setStyleSheet(f"border: 1px solid black; ")
        textLabel.setFont(QFont("Times", 24))
        layout.addWidget(textLabel)
        widget.setLayout(layout)
        return widget

    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SingleID(imageFolder="../Annotation/images_annotation_1", ID="012")
    window.show()
    sys.exit(app.exec_())