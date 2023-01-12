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
from imageGridLayout import ImageGrid
from verification import *


class SearchEngine(QWidget):
    def __init__(self, imageFolder):
        super().__init__()
        self.width = 1920
        self.height = 1920
        self.inputPaneWidth = 600
        self.imageFolder = imageFolder
        self.imgIdx = 0
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
        self.prevBtn.setFixedSize(self.inputPaneWidth // 2.1, 50)
        self.prevBtn.setFont(QFont("Times", config.ImageDirLabelSize))
        self.prevBtn.clicked.connect(self.prevBtnAction)
        self.nextBtn = QPushButton("Next")
        self.nextBtn.setFixedSize(self.inputPaneWidth // 2.1, 50)
        self.nextBtn.setFont(QFont("Times", config.ImageDirLabelSize))
        self.nextBtn.clicked.connect(self.nextBtnAction)

        self.searchBtn = QPushButton("Search")
        self.searchBtn.setFixedSize(self.inputPaneWidth // 1.1, 50)
        self.searchBtn.setFont(QFont("Times", config.ImageDirLabelSize))
        self.searchBtn.clicked.connect(self.processSingleImage)

        self.navLayout.addWidget(self.prevBtn)
        self.navLayout.addWidget(self.nextBtn)

        self.inputLayout.addLayout(self.navLayout)
        self.inputLayout.addLayout(self.inputDispLayout)
        self.inputLayout.addWidget(self.searchBtn)

        self.mainLayout.addLayout(self.inputLayout)
        self.mainLayout.addLayout(self.resultLayout)
        self.setLayout(self.mainLayout)

        self.PMXrays = self.getAll_PM_Xrays(self.imageFolder, phase="PM")
        self.AMXrays = self.getAll_PM_Xrays(self.imageFolder, phase="AM")

        print(osp.join(self.imageFolder, self.PMXrays[self.imgIdx]))

        widget = SingleID.getImageWidget(osp.join(self.imageFolder, self.PMXrays[self.imgIdx]))
        widget.setFixedWidth(512)
        self.inputDispLayout.addWidget(widget)

        result_dummy = QLabel("")
        self.resultLayout.addWidget(result_dummy)
        self.ToggleButtonActivation()
        self.model = loadModel()


    def processSingleImage(self):
        results = getTopKPredictions(self.model,
                                     probe=self.PMXrays[self.imgIdx],
                                     galleryDir="/home/sonymd/Desktop/IFACTS/IFACTS-SPG-Radiography/Annotation/images_annotation_1/",
                                     K=10)
        print(results)
        for i in reversed(range(self.resultLayout.count())):
            widgetToRemove = self.resultLayout.itemAt(i).widget()
            # remove it from the layout list
            self.resultLayout.removeWidget(widgetToRemove)
            # remove it from the gui
            widgetToRemove.setParent(None)
        self.resultLayout.addWidget(ImageGrid(results=results))

    def nextBtnAction(self):
        self.imgIdx += 1
        self.ToggleButtonActivation()
        self.updateInputDisplayImage()

    def prevBtnAction(self):
        self.imgIdx -= 1
        self.ToggleButtonActivation()
        self.updateInputDisplayImage()

    def updateInputDisplayImage(self):
        newWidget = SingleID.getImageWidget(osp.join(self.imageFolder, self.PMXrays[self.imgIdx]))
        newWidget.setFixedWidth(512)
        oldWidget = self.inputDispLayout.itemAt(0).widget()
        self.inputDispLayout.replaceWidget(oldWidget, newWidget)
        oldWidget.setParent(None)

        if self.imgIdx == len(self.PMXrays) - 1:
            self.nextBtn.setEnabled(False)

    def ToggleButtonActivation(self):
        if self.imgIdx > 0:
            self.prevBtn.setEnabled(True)
        else:
            self.prevBtn.setEnabled(False)
        if self.imgIdx < len(self.PMXrays) - 1:
            self.nextBtn.setEnabled(True)
        else:
            self.nextBtn.setEnabled(False)

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
