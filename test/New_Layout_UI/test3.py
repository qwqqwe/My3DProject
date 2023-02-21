import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from My_Setting_UI import Ui_MainWindow
from PySide2.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient)
from PySide2.QtWidgets import *
from ui_functions import *
import time
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import open3d as o3d
import numpy as np
from pathlib import Path


from vtkmodules.vtkIOImage import (
    vtkBMPWriter,
    vtkJPEGWriter,
    vtkPNGWriter,
    vtkPNMWriter,
    vtkPostScriptWriter,
    vtkTIFFWriter
)
from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter

def write_image(file_name, ren_win, rgba=True):

    if file_name:
        valid_suffixes = ['.bmp', '.jpg', '.png', '.pnm', '.ps', '.tiff']
        # Select the writer to use.
        parent = Path(file_name).resolve().parent
        path = Path(parent) / file_name
        if path.suffix:
            ext = path.suffix.lower()
        else:
            ext = '.png'
            path = Path(str(path)).with_suffix(ext)
        if path.suffix not in valid_suffixes:
            print(f'No writer for this file suffix: {ext}')
            return
        if ext == '.bmp':
            writer = vtkBMPWriter()
        elif ext == '.jpg':
            writer = vtkJPEGWriter()
        elif ext == '.pnm':
            writer = vtkPNMWriter()
        elif ext == '.ps':
            if rgba:
                rgba = False
            writer = vtkPostScriptWriter()
        elif ext == '.tiff':
            writer = vtkTIFFWriter()
        else:
            writer = vtkPNGWriter()

        windowto_image_filter = vtkWindowToImageFilter()
        windowto_image_filter.SetInput(ren_win)
        windowto_image_filter.SetScale(1)  # image quality
        if rgba:
            windowto_image_filter.SetInputBufferTypeToRGBA()
        else:
            windowto_image_filter.SetInputBufferTypeToRGB()
            # Read from the front buffer.
            windowto_image_filter.ReadFrontBufferOff()
            windowto_image_filter.Update()

        writer.SetFileName(path)
        writer.SetInputConnection(windowto_image_filter.GetOutputPort())
        writer.Write()
    else:
        raise RuntimeError('Need a filename.')


class mywindow(QtWidgets.QMainWindow,Ui_MainWindow):

    def __init__(self):
        super(mywindow,self).__init__()
        self.setupUi(self)
        self.setWindowTitle('test_gui')
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.btn_minimize.clicked.connect(self.showMinimized)
        self.btn_maximize_restore.clicked.connect(self.max_recv)
        self.btn_close.clicked.connect(self.close)
        self._padding = 5  # 设置边界宽度为5
        self.initDrag() # 设置鼠标跟踪判断默认值
        self._tracking = False
        ## ==> TOGGLE MENU SIZE
        self.btn_toggle_menu.clicked.connect(lambda: UIFunctions.toggleMenu(self, 220, True))#设置动画
        UIFunctions.selectStandardMenu(self, "btn_running")
        self.stackedWidget.setCurrentWidget(self.page_home)
        self.btn_running.clicked.connect(self.Button)
        self.btn_settings.clicked.connect(self.Button)
        #vtk設置
        self.frame = QtWidgets.QFrame()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)

        self.cloud_point.addWidget(self.vtkWidget)
        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        #####這行很重要，因為vtk有兩種交互方式，一種是trackball模式，還有一種是joystick模式，trackball模式才和open3d的操作模式一樣
        #####joystick模式下的操作一般人做不來
        # Create source
        txt_path = '../../txtcouldpoint/Finalzhengzheng5.txt'
        pcd = np.loadtxt(txt_path, delimiter=",")

        poins = vtk.vtkPoints()
        for i in range(pcd.shape[0]):
            dp = pcd[i]
            poins.InsertNextPoint(dp[0], dp[1], dp[2])
        a=time.time()
        Temp_Mid_X=(np.max(pcd[:,0])+np.min(pcd[:,0]))/2
        Temp_Mid_Y=(np.max(pcd[:,1])+np.min(pcd[:,1]))/2
        Temp_Mid_Z=(np.max(pcd[:,2])+np.min(pcd[:,2]))/2
        b=time.time()
        print(b-a)
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(poins)

        glyphFilter = vtk.vtkVertexGlyphFilter()
        glyphFilter.SetInputData(polydata)
        glyphFilter.Update()

        dataMapper = vtk.vtkPolyDataMapper()
        dataMapper.SetInputConnection(glyphFilter.GetOutputPort())

        # Create an actor
        actor = vtk.vtkActor()
        actor.SetMapper(dataMapper)
        self.ren.AddActor(actor)
        #这是上视图
        camera = self.ren.GetActiveCamera()
        self.up_view(camera,Temp_Mid_X,Temp_Mid_Y,Temp_Mid_Z)
        #这是显示我们的坐标轴的
        axesActor = vtk.vtkAxesActor()
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(axesActor)
        self.axes_widget.SetInteractor(self.iren)
        self.axes_widget.EnabledOn()
        self.axes_widget.InteractiveOff()

        self.show()
        self.iren.Initialize()

        self.btn_down_view.clicked.connect(lambda: self.up_view(camera, Temp_Mid_X, Temp_Mid_Y, Temp_Mid_Z))
        self.btn_left_view.clicked.connect(lambda: self.left_view(camera, Temp_Mid_X, Temp_Mid_Y, Temp_Mid_Z))

    def Button(self):
        # GET BT CLICKED
        btnWidget = self.sender()
        # PAGE HOME
        if btnWidget.objectName() == "btn_running":
            self.stackedWidget.setCurrentWidget(self.page_home)
            UIFunctions.resetStyle(self, "btn_running")
            UIFunctions.labelPage(self, "运行")
            btnWidget.setStyleSheet(UIFunctions.selectMenu(btnWidget.styleSheet()))

        if btnWidget.objectName() == "btn_settings":
            self.stackedWidget.setCurrentWidget(self.page_settings)
            UIFunctions.resetStyle(self, "btn_settings")
            UIFunctions.labelPage(self, "设置")
            btnWidget.setStyleSheet(UIFunctions.selectMenu(btnWidget.styleSheet()))

    def initDrag(self):
        # 设置鼠标跟踪判断扳机默认值
        self._move_drag = False
        self._corner_drag = False
        self._bottom_drag = False
        self._right_drag = False

    def resizeEvent(self, QResizeEvent):
        # 重新调整边界范围以备实现鼠标拖放缩放窗口大小，采用三个列表生成式生成三个列表
        self._right_rect = [QtCore.QPoint(x, y) for x in range(self.centralwidget.width() - self._padding, self.centralwidget.width() + 1)
                            for y in range(1, self.centralwidget.height() - self._padding)]
        self._bottom_rect = [QtCore.QPoint(x, y) for x in range(1, self.centralwidget.width() - self._padding)
                             for y in range(self.centralwidget.height() - self._padding, self.centralwidget.height() + 1)]
        self._corner_rect = [QtCore.QPoint(x, y) for x in range(self.centralwidget.width() - self._padding, self.centralwidget.width() + 1)
                             for y in range(self.centralwidget.height() - self._padding, self.centralwidget.height() + 1)]

    def max_recv(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):  # 重写移动事件
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

    def up_view(self,camera,Temp_Mid_X,Temp_Mid_Y,Temp_Mid_Z):
        camera.SetViewUp(0, 1, 0)
        camera.SetPosition(Temp_Mid_X, Temp_Mid_Y, Temp_Mid_Z+250)
        camera.SetFocalPoint(Temp_Mid_X, Temp_Mid_Y, Temp_Mid_Z)
        self.ren.ResetCameraClippingRange()
        self.show()
        self.iren.Initialize()

    # 左視圖函數
    def left_view(self,camera,Temp_Mid_X,Temp_Mid_Y,Temp_Mid_Z):
        camera.SetViewUp(0, 0, 1)
        camera.SetPosition(Temp_Mid_X, Temp_Mid_Y-200, Temp_Mid_Z)
        camera.SetFocalPoint(Temp_Mid_X, Temp_Mid_Y, Temp_Mid_Z)
        self.ren.ResetCameraClippingRange()
        self.show()
        self.iren.Initialize()

    def saveToimage(self):
        from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter
        from vtkmodules.vtkIOImage import (
            vtkBMPWriter, vtkJPEGWriter, vtkPNGWriter,
            vtkPNMWriter, vtkPostScriptWriter, vtkTIFFWriter
        )
        # 初始化
        writer = vtkBMPWriter()
        # writer = vtkPostScriptWriter()
        windowto_image_filter = vtkWindowToImageFilter()
        windowto_image_filter.SetInput(self.vtkWidget.GetRenderWindow())  # 你渲染窗口中的图片
        windowto_image_filter.SetScale(5)                                  # 图像尺寸
        windowto_image_filter.SetInputBufferTypeToRGBA()                   # 四通道
        # 保存
        writer.SetFileName('cs.jpg')
        # writer.SetFileName('cs.ps')
        writer.SetInputConnection(windowto_image_filter.GetOutputPort())
        writer.Write()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myshow = mywindow()
    myshow.show()
    sys.exit(app.exec_())
