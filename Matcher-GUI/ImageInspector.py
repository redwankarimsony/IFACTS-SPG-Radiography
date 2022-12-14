# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Inspector.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!

def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = x - (w / 2)
    xmax = x + (w / 2)
    ymin = y - (h / 2)
    ymax = y + (h / 2)
    return xmin, ymin, xmax, ymax

import glob
import cv2
import os.path as osp
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1280, 720)
        self.currentIDx = 0
        self.imgFiles = self.scanAllFiles("../Annotation/images_annotation_1")

        self.imgLabel = QtWidgets.QLabel(Form)
        self.imgLabel.setGeometry(QtCore.QRect(80, 20, 981, 591))
        self.imgLabel.setText("")
        self.imgLabel.setObjectName("imgLabel")
        self.imgLabel.setAlignment(Qt.AlignCenter)
        self.imgName = QtWidgets.QLabel(Form)
        self.imgName.setGeometry(QtCore.QRect(450, 650, 301, 60))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.imgName.setFont(font)
        self.imgName.setAlignment(QtCore.Qt.AlignCenter)
        self.imgName.setObjectName("imgName")
        self.prevBtn = QtWidgets.QPushButton(Form)
        self.prevBtn.setGeometry(QtCore.QRect(260, 650, 120, 60))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.prevBtn.setFont(font)
        self.prevBtn.setObjectName("prevBtn")
        self.prevBtn.clicked.connect(self.prevBtnAction)
        self.nextBtn = QtWidgets.QPushButton(Form)
        self.nextBtn.setGeometry(QtCore.QRect(790, 650, 120, 60))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.nextBtn.setFont(font)
        self.nextBtn.setObjectName("nextBtn")
        self.nextBtn.clicked.connect(self.nextBtnAction)
        self.checkBox = QtWidgets.QCheckBox(Form)
        self.checkBox.setGeometry(QtCore.QRect(990, 650, 151, 60))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.checkBox.setFont(font)
        self.checkBox.setIconSize(QtCore.QSize(32, 32))
        self.checkBox.setObjectName("checkBox")

        self.retranslateUi(Form)
        self.imgName.setText(self.imgFiles[self.currentIDx].split("/")[-1])
        QtCore.QMetaObject.connectSlotsByName(Form)

    @staticmethod
    def scanAllFiles(imageFolder):
        imgFiles = []
        imgFormats = ["jpg", "png", "jpeg", "bmp"]
        for imgFormat in imgFormats:
            imgFiles.extend(glob.glob(osp.join(imageFolder, f"*.{imgFormat}")))
            imgFiles.extend(glob.glob(osp.join(imageFolder, f"*.{imgFormat.upper()}")))

        return sorted(imgFiles)

    @staticmethod
    def drawLabel(imgPath):
        img = cv2.imread(imgPath)
        annotPath = imgPath.replace(imgPath.split(".")[-1], "txt")
        if osp.exists(annotPath):
            with open(annotPath, "r") as f:
                line= f.readline()

            [label, x, y, width, height] = line.split(" ")
            
            [xmin, ymin, xmax, ymax] = bbox2points([float(x), float(y), float(width), float(height)])
            H, W, C = img.shape
            start_point = (int(xmin*W), int(ymin*H))
            end_point = (int(xmax*W), int(ymax*H))
            
            print(start_point, end_point)
            color = (255, 0, 0)
            thickness = 2
            newImg = cv2.rectangle(img.copy(), start_point, end_point, color, thickness)


            height, width, channel = newImg.shape
            bytesPerLine = 3 * width
            qImg = QImage(newImg.data, width, height, bytesPerLine, QImage.Format_RGB888)
            return qImg
        else:
            height, width, channel = img.shape
            bytesPerLine = 3 * width
            qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            return qImg
        



    def nextBtnAction(self):
        if self.currentIDx+1 != len(self.imgFiles):
            self.currentIDx+=1
            self.imgName.setText(self.imgFiles[self.currentIDx].split("/")[-1])
            imgPath = self.imgFiles[self.currentIDx]
            if not self.checkBox.isChecked():
                self.imgLabel.setPixmap(QPixmap(imgPath).scaled(720, 720, Qt.KeepAspectRatio))
            else:


                self.imgLabel.setPixmap(QPixmap(self.drawLabel(imgPath)).scaled(720, 720, Qt.KeepAspectRatio))
 



    def prevBtnAction(self):
            if self.currentIDx-1 >0:
                self.currentIDx-=1
                self.imgName.setText(self.imgFiles[self.currentIDx].split("/")[-1])
                imgPath = self.imgFiles[self.currentIDx]
                if not self.checkBox.isChecked():
                    self.imgLabel.setPixmap(QPixmap(imgPath).scaled(720, 720, Qt.KeepAspectRatio))
                else:
                    self.imgLabel.setPixmap(QPixmap(self.drawLabel(imgPath)).scaled(720, 720, Qt.KeepAspectRatio))



    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.imgName.setText(_translate("Form", "TextLabel"))
        self.prevBtn.setText(_translate("Form", "Prev"))
        self.nextBtn.setText(_translate("Form", "Next"))
        self.checkBox.setText(_translate("Form", "Annotation"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
