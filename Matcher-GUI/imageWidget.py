# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'form.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(640, 480)
        self.imgLabel = QtWidgets.QLabel(Form)
        self.imgLabel.setGeometry(QtCore.QRect(20, 30, 600, 400))
        self.imgLabel.setText("")
        self.imgLabel.setObjectName("imgLabel")
        self.imgName = QtWidgets.QLabel(Form)
        self.imgName.setGeometry(QtCore.QRect(170, 435, 321, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.imgName.setFont(font)
        self.imgName.setAlignment(QtCore.Qt.AlignCenter)
        self.imgName.setObjectName("imgName")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.imgName.setText(_translate("Form", "TextLabel"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
