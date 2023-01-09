import os
import sys
import os.path as osp
import glob
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QHBoxLayout, QDesktopWidget, \
    QGridLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt
from config import config
from imageDisplayWidget import SingleID


class SearchEngine(QWidget):
    def __init__(self, imageFolder):
        super().__init__()
        self.imageFolder = imageFolder
        sizeObject = QDesktopWidget().screenGeometry(-1)
        self.setWindowIcon(QIcon(config.appIcon))
        self.setFixedSize(int(sizeObject.width() * .7), int(sizeObject.height() * .7))

        self.mainLayout = QHBoxLayout()
        self.inputLayout = QVBoxLayout()
        self.inputDispLayout = QVBoxLayout()
        self.resultLayout = QGridLayout()

        # Making the Navigation layout
        self.navLayout = QHBoxLayout()
        self.prevBtn = QPushButton("Prev")
        self.prevBtn.setFixedHeight(50)
        self.prevBtn.setFont(QFont("Times", config.ImageDirLabelSize))
        self.nextBtn = QPushButton("Next")
        self.nextBtn.setFixedHeight(50)
        self.nextBtn.setFont(QFont("Times", config.ImageDirLabelSize))

        self.navLayout.addWidget(self.prevBtn)
        self.navLayout.addWidget(self.nextBtn)

        self.inputLayout.addLayout(self.navLayout)
        self.inputLayout.addLayout(self.inputDispLayout)

        self.mainLayout.addLayout(self.inputLayout)
        self.mainLayout.addLayout(self.resultLayout)
        self.setLayout(self.mainLayout)

        self.PMXrays = self.getAll_PM_Xrays(self.imageFolder, phase="PM")
        self.AMXrays = self.getAll_PM_Xrays(self.imageFolder, phase="AM")

        print(osp.join(self.imageFolder, self.PMXrays[0]))
        widget = SingleID.getImageWidget(osp.join(self.imageFolder, self.PMXrays[0]))
        widget.setFixedWidth(800)
        self.inputDispLayout.addWidget(widget)


        self.nextBtn.clicked.connect(self.openFileNameDialog)

    @staticmethod
    def getAll_PM_Xrays(imgDir, phase="PM"):
        frmts = ["JPG", "PNG", "JPEG", "BMP", "png", "jpg", "jpeg", "bmp"]
        validImageFiles = []
        if osp.exists(imgDir):
            files = os.listdir(imgDir)
            for filePath in files:
                ext = filePath.split(".")[-1]
                if (ext in frmts) and (phase in filePath):
                    validImageFiles.append(filePath)
            return validImageFiles
        else:
            print("ERROR: Image Directory does not exists!")
            return []

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,
                                                  "QFileDialog.getOpenFileName()",
                                                  "",
                                                  "All Files (*);;Python Files (*.py)",
                                                  options=options)
        if fileName:
            print(fileName)

    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self,
                                                "QFileDialog.getOpenFileNames()",
                                                "",
                                                "All Files (*);;Python Files (*.py)",
                                                options=options)
        if files:
            print(files)

    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,
                                                  "QFileDialog.getSaveFileName()",
                                                  "",
                                                  "All Files (*);;Text Files (*.txt)",
                                                  options=options)
        if fileName:
            print(fileName)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SearchEngine(imageFolder=config.imageDir)
    # window.setGeometry(10, 10, 1920, 800)
    window.show()
    sys.exit(app.exec_())
