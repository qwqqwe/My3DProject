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


        # kkk=self.centralwidget.parent().objectName()

        # self.setMouseTracking(True)
        # self.centralwidget.setMouseTracking(True)
        # self.setMouseTracking(True)

        # QtWidgets.QMainWindow.setMouseTracking(QtWidgets.QMainWindow,True)
        # self.setMouseTracking(True) # 设置widget鼠标跟踪
        # self.setCentralWidget(self.MainWindow)
        # self.MainWindow.setMouseTracking(True)


        self._padding = 5  # 设置边界宽度为5
        self.initDrag() # 设置鼠标跟踪判断默认值
        self._tracking = False

    def initDrag(self):
        # 设置鼠标跟踪判断扳机默认值
        self._move_drag = False
        self._corner_drag = False
        self._bottom_drag = False
        self._right_drag = False
    def resizeEvent(self, QResizeEvent):
        # print(123)
        a=self.centralwidget.width()
        b=self.centralwidget.height()
        c=self._padding


        # 重新调整边界范围以备实现鼠标拖放缩放窗口大小，采用三个列表生成式生成三个列表
        self._right_rect = [QtCore.QPoint(x, y) for x in range(self.centralwidget.width() - self._padding, self.centralwidget.width() + 1)
                            for y in range(1, self.centralwidget.height() - self._padding)]
        self._bottom_rect = [QtCore.QPoint(x, y) for x in range(1, self.centralwidget.width() - self._padding)
                             for y in range(self.centralwidget.height() - self._padding, self.centralwidget.height() + 1)]
        self._corner_rect = [QtCore.QPoint(x, y) for x in range(self.centralwidget.width() - self._padding, self.centralwidget.width() + 1)
                             for y in range(self.centralwidget.height() - self._padding, self.centralwidget.height() + 1)]
        print(self._right_rect)
    def max_recv(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()


    def mouseMoveEvent(self, e: QtGui.QMouseEvent):  # 重写移动事件
        # a=self._tracking
        # print(a)
        # print(e.pos())
        # print(self._right_rect)
        if self._tracking:
            self._endPos = e.pos() - self._startPos
            self.move(self.pos() + self._endPos)
        if e.pos() in self._corner_rect:
            self.setCursor(QtCore.Qt.SizeFDiagCursor)

        elif e.pos() in self._bottom_rect:
            self.setCursor(QtCore.Qt.SizeVerCursor)
        elif e.pos() in self._right_rect:
            self.setCursor(QtCore.Qt.SizeHorCursor)
        else:
            self.setCursor(QtCore.Qt.ArrowCursor)
            # 当鼠标左键点击不放及满足点击区域的要求后，分别实现不同的窗口调整
            # 没有定义左方和上方相关的5个方向，主要是因为实现起来不难，但是效果很差，拖放的时候窗口闪烁，再研究研究是否有更好的实现
        if QtCore.Qt.LeftButton and self._right_drag:
            # 右侧调整窗口宽度
            self.resize(e.pos().x(), self.height())
            e.accept()
        elif QtCore.Qt.LeftButton and self._bottom_drag:
            # 下侧调整窗口高度
            self.resize(self.width(), e.pos().y())
            e.accept()
        elif QtCore.Qt.LeftButton and self._corner_drag:
            # 右下角同时调整高度和宽度
            self.resize(e.pos().x(), e.pos().y())
            e.accept()

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        a=self.centralwidget.height()
        b=self.centralwidget.width()
        print(a)
        if (e.button() == QtCore.Qt.LeftButton) and (self.frame_top_btns.underMouse()):
            self._startPos = QtCore.QPoint(e.x(), e.y())
            self._tracking = True

        if (e.button() == QtCore.Qt.LeftButton) and (e.pos() in self._corner_rect):
            # 鼠标左键点击右下角边界区域
            self._corner_drag = True
            e.accept()
        elif (e.button() == QtCore.Qt.LeftButton) and (e.pos() in self._right_rect):
            # 鼠标左键点击右侧边界区域
            self._right_drag = True
            e.accept()
        elif (e.button() == QtCore.Qt.LeftButton) and (e.pos() in self._bottom_rect):
            # 鼠标左键点击下侧边界区域
            self._bottom_drag = True
            e.accept()


    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.LeftButton:
            self._tracking = False
            self._startPos = None
            self._endPos = None
        self._move_drag = False
        self._corner_drag = False
        self._bottom_drag = False
        self._right_drag = False

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myshow = mywindow()
    myshow.show()
    sys.exit(app.exec_())
