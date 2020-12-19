import os

from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap


def clickable(widget):
    class Filter(QtWidgets.QWizard):
        clicked = pyqtSignal()

        def eventFilter(self, obj, event):
            if obj == widget:
                if event.type() == QEvent.MouseButtonRelease:
                    if obj.rect().contains(event.pos()):
                        self.clicked.emit()
                        return True
            return False

    filter = Filter(widget)
    widget.installEventFilter(filter)
    return filter.clicked


class Ui(QtWidgets.QWizard):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi("style.ui", self)

        self.pushButton.clicked.connect(self.inputUserPhoto)
        clickable(self.label_5).connect(self.inputUserStyle)
        clickable(self.label_style1).connect(self.inputStyle1)
        clickable(self.label_style2).connect(self.inputStyle2)
        clickable(self.label_style3).connect(self.inputStyle3)

        self.startTf.clicked.connect(self.doAction)
        self.progressBar.setMaximum(100)
        self.timer = QBasicTimer()
        self.step = 0

    def timerEvent(self, e):

        if self.step >= 100:
            self.timer.stop()
            self.startTf.setText('Сохранить изображение')
            self.startTf.clicked(self.outputImage)
            return

        self.step = self.step + 0.01
        self.progressBar.setValue(self.step)


    def doAction(self):

        if self.timer.isActive():
            self.timer.stop()
            self.startTf.setText('Начать обработку')
        else:
            self.timer.start(100, self)
            self.startTf.setText('Прервать обработку')

    def outputImage(self):
        style_reference_image_path = "my_result_at_iteration_0.png"

    def inputUserPhoto(self):
        fileName = QtWidgets.QFileDialog.getOpenFileName(self, "Выбор изображения", "/", "Image Files (*.png *.jpg)")
        self.lineEdit.setText(fileName[0])
        userfoto = QPixmap(fileName[0]).scaled(300, 200)
        self.label_2.setPixmap(userfoto)
        target_image_path = fileName[0]
        # alert = QtWidgets.QMessageBox()
        # alert.setText(fileName[0])
        # alert.exec_()

    def inputUserStyle(self):
        fileName = QtWidgets.QFileDialog.getOpenFileName(self, "Выбор изображения", "/", "Image Files (*.png *.jpg)")
        userfoto = QPixmap(fileName[0]).scaled(300, 200)
        self.label_5.setPixmap(userfoto)
        style_reference_image_path = fileName[0]

    def inputStyle1(self):
        style_reference_image_path = "img/s1.jpg"
        self.next()

    def inputStyle2(self):
        style_reference_image_path = "img/s2.jpg"
        self.next()

    def inputStyle3(self):
        style_reference_image_path = "img/s3.jpg"
        self.next()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    window.show()
    sys.exit(app.exec_())
