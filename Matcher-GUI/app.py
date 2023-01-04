import sys
import os.path as osp
import glob
from imageDisplayWidget import SingleID as SingleIDImageViewer
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QHBoxLayout, QDesktopWidget
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt
from config import config



class ImageViewer(QWidget):
    def __init__(self, imageFolder):
        super().__init__()
        sizeObject = QDesktopWidget().screenGeometry(-1)
        self.setWindowIcon(QIcon(config.appIcon))
        self.setFixedSize(int(sizeObject.width()*.7), int(sizeObject.height()*.7))
        self.IDs = self.getAllIDs(imageFolder=imageFolder)
        self.currentIDx = 0
        self.imageFolder = imageFolder

        self.mainLayout = QVBoxLayout()

        # Making the Navigation layout
        self.navLayout = QHBoxLayout()
        self.prevBtn = QPushButton("Prev")
        self.prevBtn.setFixedHeight(50)
        self.prevBtn.setFont(QFont("Times", config.ImageDirLabelSize))
        self.nextbtn = QPushButton("Next")
        self.nextbtn.setFixedHeight(50)
        self.nextbtn.setFont(QFont("Times", config.ImageDirLabelSize))

        self.prevBtn.clicked.connect(self.prevBtnAction)
        self.nextbtn.clicked.connect(self.nextBtnAction)

        self.idLabel = QLabel("ID: " + self.IDs[self.currentIDx])
        self.idLabel.setAlignment(Qt.AlignCenter)
        self.idLabel.setFixedHeight(50)
        self.idLabel.setFont(QFont("Times", config.ImageDirLabelSize))

        self.navLayout.addWidget(self.prevBtn)
        self.navLayout.addWidget(self.idLabel)
        self.navLayout.addWidget(self.nextbtn)

        self.imgWidget = SingleIDImageViewer(imageFolder=self.imageFolder, ID=self.IDs[self.currentIDx])
        self.mainLayout.addLayout(self.navLayout)
        self.mainLayout.addWidget(self.imgWidget)

        self.imageDirLabel = QLabel(f"ImageFolder: {imageFolder}")
        self.imageDirLabel.setFont(QFont("Times", config.ImageDirLabelSize))
        self.imageDirLabel.setFixedSize(config.ImageDirLabelWidth, config.ImageDirLabelHeight)
        self.imageDirLabel.setAlignment(Qt.AlignCenter)

        self.mainLayout.addWidget(self.imageDirLabel)


        self.setLayout(self.mainLayout)
        self.center()

    def nextBtnAction(self):
        if self.currentIDx + 1 != len(self.IDs):
            self.currentIDx += 1
            self.idLabel.setText("ID: " + self.IDs[self.currentIDx])
            self.replaceImageLayout()

    def prevBtnAction(self):
        if self.currentIDx - 1 > 0:
            self.currentIDx -= 1
            self.idLabel.setText("ID: " + self.IDs[self.currentIDx])
            self.replaceImageLayout()

    def replaceImageLayout(self):
        imgWidgetOld = self.mainLayout.itemAt(1).widget()

        self.imgWidget = SingleIDImageViewer(imageFolder=self.imageFolder, ID=self.IDs[self.currentIDx])
        # self.mainLayout.addWidget(self.imgWidget)
        self.mainLayout.replaceWidget(imgWidgetOld, self.imgWidget)
        imgWidgetOld.setParent(None)

    @staticmethod
    def getAllIDs(imageFolder):
        """This method produces all the unique IDs used in the dataset."""
        imgFiles = []
        imgFormats = ["jpg", "png", "jpeg", "bmp"]
        for imgFormat in imgFormats:
            imgFiles.extend(glob.glob(osp.join(imageFolder, f"*.{imgFormat}")))
            imgFiles.extend(glob.glob(osp.join(imageFolder, f"*.{imgFormat.upper()}")))

        IDs = set()
        for imgFile in imgFiles:
            ID = imgFile.split("/")[-1].split("_")[0]
            IDs.add(ID)
        return sorted(list(IDs))

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())







if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = ImageViewer(imageFolder=config.imageDir)
    # window.setGeometry(10, 10, 1920, 800)
    window.show()
    sys.exit(app.exec_())
