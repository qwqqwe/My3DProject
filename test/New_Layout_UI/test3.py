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

from function_detect import *


from vtkmodules.vtkIOImage import (
    vtkBMPWriter,
    vtkJPEGWriter,
    vtkPNGWriter,
    vtkPNMWriter,
    vtkPostScriptWriter,
    vtkTIFFWriter
)
from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter

# windll.LoadLibrary(r"C:\Users\Administrator\Documents\WeChat Files\wxid_bj5u8pz1th8e12\FileStorage\File\2022-08\v2.1.15.138(1)\G56N_SDK_DEMO_2.1.15.138_20220121_1748\CamWrapper\bins\X64\Debug\SgCamWrapper.dll")
# targe1t=windll.LoadLibrary(r"C:\Users\Administrator\Documents\WeChat Files\wxid_bj5u8pz1th8e12\FileStorage\File\2022-08\v2.1.15.138(1)\G56N_SDK_DEMO_2.1.15.138_20220121_1748\CamWrapper\bins\X64\Debug\Dll6.dll")
# targe1t=windll.LoadLibrary(r"C:\Users\Administrator\Documents\WeChat Files\wxid_bj5u8pz1th8e12\FileStorage\File\2022-08\v2.1.15.138(1)\G56N_SDK_DEMO_2.1.15.138_20220121_1748\CamWrapper\bins\X64\Debug\Dll6.dll")

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
        self._padding = 5  # 设置边界宽度为5
        self.initDrag() # 设置鼠标跟踪判断默认值
        self._tracking = False
        ## ==> TOGGLE MENU SIZE
        self.btn_toggle_menu.clicked.connect(lambda: UIFunctions.toggleMenu(self, 220, True))#设置动画
        UIFunctions.selectStandardMenu(self, "btn_running")
        self.stackedWidget.setCurrentWidget(self.page_home)

        #按鈕連接函數設定
        self.btn_down_view.clicked.connect(lambda: self.up_view(self.camera, self.Temp_Mid_X, self.Temp_Mid_Y, self.Temp_Mid_Z))
        self.btn_left_view.clicked.connect(lambda: self.left_view(self.camera, self.Temp_Mid_X, self.Temp_Mid_Y, self.Temp_Mid_Z))
        self.btn_minimize.clicked.connect(self.showMinimized)
        self.btn_maximize_restore.clicked.connect(self.max_recv)
        self.btn_close.clicked.connect(self.close)
        self.btn_running.clicked.connect(self.Button)
        self.btn_settings.clicked.connect(self.Button)
        self.btn_connect.clicked.connect(self.Prepare_To_Catch)
        self.btn_stop.clicked.connect(self.Stop_Conneted)
        self.btn_detect.clicked.connect(self.To_Catch)

        #vtk設置
        self.frame = QtWidgets.QFrame()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)

        self.cloud_point.addWidget(self.vtkWidget)
        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.poins = vtk.vtkPoints()
        self.polydata = vtk.vtkPolyData()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkCells = vtk.vtkCellArray()
        # self.ren.SetBackground(1,1,1)
        #####這行很重要，因為vtk有兩種交互方式，一種是trackball模式，還有一種是joystick模式，trackball模式才和open3d的操作模式一樣
        #####joystick模式下的操作一般人做不來
        # Create source
        # txt_path = '../../txtcouldpoint/Finalzhengzheng5.txt'
        # self.pcd = np.loadtxt(txt_path, delimiter=",")
        self.pcd = None

        self.glyphFilter = vtk.vtkVertexGlyphFilter()
        self.glyphFilter.SetInputData(self.polydata)
        self.glyphFilter.Update()
        self.dataMapper = vtk.vtkPolyDataMapper()
        self.dataMapper.SetInputConnection(self.glyphFilter.GetOutputPort())

        # Create an actor
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.dataMapper)
        self.ren.AddActor(self.actor)
        #这是上视图
        self.camera = self.ren.GetActiveCamera()
        self.Temp_Mid_X = 0
        self.Temp_Mid_Y = 0
        self.Temp_Mid_Z = 0
        # self.up_view(self.camera,self.Temp_Mid_X,self.Temp_Mid_Y,self.Temp_Mid_Z)
        #这是显示我们的坐标轴的
        axesActor = vtk.vtkAxesActor()
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(axesActor)
        self.axes_widget.SetInteractor(self.iren)
        self.axes_widget.EnabledOn()
        self.axes_widget.InteractiveOff()

        # self.show()
        self.iren.Initialize()

    def Change_To_VTK(self):
        for i in range(self.pcd.shape[0]):
            self.poins.InsertNextPoint(self.pcd[i][0], self.pcd[i][1], self.pcd[i][2])
            self.vtkDepth.InsertNextValue(self.pcd[i][3])
        self.vtkDepth.Modified()
        self.Temp_Mid_X = (np.max(self.pcd[:, 0]) + np.min(self.pcd[:, 0])) / 2
        self.Temp_Mid_Y = (np.max(self.pcd[:, 1]) + np.min(self.pcd[:, 1])) / 2
        self.Temp_Mid_Z = (np.max(self.pcd[:, 2]) + np.min(self.pcd[:, 2])) / 2
        self.polydata.SetPoints(self.poins)
        self.polydata.SetVerts(self.vtkCells)
        self.polydata.GetPointData().SetScalars(self.vtkDepth)
    def To_Catch(self):
        # pcd = Py_Catch(targe1t)
        txt_path = '../../txtcouldpoint/Finalzhengzheng5.txt'
        pcd = np.loadtxt(txt_path, delimiter=",")
        self.pcd,defect_meassage = display2(pcd)
        self.Change_To_VTK()
        self.up_view(self.camera, self.Temp_Mid_X, self.Temp_Mid_Y, self.Temp_Mid_Z)
        self.textBrowser.setText(str(defect_meassage))

    def Stop_Conneted(self):
        a=Py_Stop(targe1t)
        return a
    def Prepare_To_Catch(self):
        a=Py_PrepareToCatch(targe1t)
        return a
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
        camera.SetPosition(Temp_Mid_X, Temp_Mid_Y, Temp_Mid_Z+200)
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
