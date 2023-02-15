import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from My_Setting_UI import Ui_MainWindow

class mywindow(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(mywindow,self).__init__()
        self.setupUi(self)
        self.setWindowTitle('test_gui')
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.btn_minimize.clicked.connect(self.showMinimized)
        self.btn_maximize_restore.clicked.connect(self.max_recv)
        self.btn_close.clicked.connect(self.close)


    def max_recv(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):  # 重写移动事件
        if self._tracking:
            self._endPos = e.pos() - self._startPos
            self.move(self.pos() + self._endPos)

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if (e.button() == QtCore.Qt.LeftButton) and (self.frame_top_btns.underMouse()):
            self._startPos = QtCore.QPoint(e.x(), e.y())
            self._tracking = True

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.LeftButton:
            self._tracking = False
            self._startPos = None
            self._endPos = None

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myshow = mywindow()
    myshow.show()
    sys.exit(app.exec_())
