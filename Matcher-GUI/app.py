import os.path as osp
import glob
from demo import SingleID as SingleIDImageViewer
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import sys



class ImageViewer(QWidget):
    def __init__(self, imageFolder):
        super().__init__()
        self.IDs = self.getAllIDs(imageFolder=imageFolder)
        self.currrentIDx = 0 

        mainLayout = QVBoxLayout()

        # Making the Navigation layout
        self.navLayout = QHBoxLayout()

        self.prevBtn = QPushButton("Prev")
        self.nextbtn = QPushButton("Next")
        self.idLabel = QLabel(self.IDs[self.currrentIDx])

        self.navLayout.addWidget(self.prevBtn)
        self.navLayout.addWidget(self.idLabel)
        self.navLayout.addWidget(self.nextbtn)


        mainLayout.addLayout(self.navLayout)
        self.setLayout(mainLayout)

    def nextBtnAction(self):
        if self.currentIDx+1 != len(self.IDs):
            self.currrentIDx+=1
            self.idLabel = QLabel(self.IDs[self.currrentIDx])

    def prevBtnAction(self):
            if self.currentIDx-1 >0:
                self.currrentIDx-=1
                self.idLabel = QLabel(self.IDs[self.currrentIDx])

    
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
            print(imgFile)
            ID = imgFile.split("/")[-1].split("_")[0]
            IDs.add(ID)
        return sorted(list(IDs))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageViewer(imageFolder="../Annotation/images_annotation_1")
    window.show()
    sys.exit(app.exec_())
        

