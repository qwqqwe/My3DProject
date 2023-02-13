import time

from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import vtk
from PyQt5 import QtCore, QtGui, QtWidgets
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from untitled import Ui_MainWindow
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
    """
    Write the render window view to an image file.

    Image types supported are:
     BMP, JPEG, PNM, PNG, PostScript, TIFF.
    The default parameters are used for all writers, change as needed.

    :param file_name: The file name, if no extension then PNG is assumed.
    :param ren_win: The render window.
    :param rgba: Used to set the buffer type.
    :return:
    """

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




class Mywindow(QtWidgets.QMainWindow, Ui_MainWindow):

    #上視圖函數
    def up_view(self,camera,Temp_Mid_X,Temp_Mid_Y,Temp_Mid_Z):
        print(111)
        camera.SetViewUp(0, 1, 0)
        camera.SetPosition(Temp_Mid_X, Temp_Mid_Y, Temp_Mid_Z+250)
        camera.SetFocalPoint(Temp_Mid_X, Temp_Mid_Y, Temp_Mid_Z)
        self.ren.ResetCameraClippingRange()
        self.show()
        self.iren.Initialize()

    # 左視圖函數
    def left_view(self,camera,Temp_Mid_X,Temp_Mid_Y,Temp_Mid_Z):
        print(222)
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


    def __init__(self):
        super(Mywindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('test_gui')

        self.frame = QtWidgets.QFrame()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.formLayout.addWidget(self.vtkWidget)

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
        # print(Temp_Mid_X,Temp_Mid_Y,Temp_Mid_Z)
        # Temp_Mid=pcd[pcd.shape[0]//2]


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

        # actor1 = vtk.vtkActor()
        # actor1.SetMapper(dataMapper)

        self.ren.AddActor(actor)
        # self.ren.AddActor(actor1)

        # self.ren.ResetCamera()

        #这是上视图
        camera = self.ren.GetActiveCamera()
        self.up_view(camera,Temp_Mid_X,Temp_Mid_Y,Temp_Mid_Z)
        # self.left_view(camera,Temp_Mid_X,Temp_Mid_Y,Temp_Mid_Z)
        # camera.SetViewUp(1, 0, 0)
        # camera.SetPosition(Temp_Mid_X, Temp_Mid_Y, Temp_Mid_Z+250)
        # camera.SetFocalPoint(Temp_Mid_X, Temp_Mid_Y, Temp_Mid_Z)

        # #这是左视图
        # camera.SetViewUp(0, 0, 1)
        # camera.SetPosition(Temp_Mid_X, Temp_Mid_Y-200, Temp_Mid_Z)
        # camera.SetFocalPoint(Temp_Mid_X, Temp_Mid_Y, Temp_Mid_Z)

        #这是显示我们的坐标轴的
        axesActor = vtk.vtkAxesActor()
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(axesActor)
        self.axes_widget.SetInteractor(self.iren)
        self.axes_widget.EnabledOn()
        self.axes_widget.InteractiveOff()



        # camera.ComputeViewPlaneNormal()
        # 設置視角
        # camera.SetPosition(-500, 0, 500)# 相機位置
        # camera.SetFocalPoint(0, 0, 0)# 焦點位置
        # camera.SetViewUp()
        # 設置物體
        # camera.Azimuth(60)# 表示 camera 的视点位置沿顺时针旋转 60 度角
        # camera.Elevation(-300)# 表示 camera 的视点位置沿向上的方面旋转 -300 度角
        # camera.Dolly(1.5)# Dolly()方法沿着视平面法向移动相机，实现放大或缩小可见角色物体
        # camera.Pitch(11)# Maybe是俯仰操作

        # self.frame.setLayout(self.formLayout)
        # self.setCentralWidget(self.frame)

        self.show()
        self.iren.Initialize()

        self.pushButton.clicked.connect(lambda: self.up_view(camera, Temp_Mid_X, Temp_Mid_Y, Temp_Mid_Z))
        self.pushButton_2.clicked.connect(lambda: self.left_view(camera, Temp_Mid_X, Temp_Mid_Y, Temp_Mid_Z))



        # write_image('E:/abcd.bmp',self.vtkWidget.GetRenderWindow())#這個函數圖片小，但是質量差
        # saveToimage這個函數圖片大，但是質量並沒有符合圖片的大小
        # bmp格式:标准的位图格式，缺点是完全不压缩，体积极大，且一旦压缩有可能掉颜色，优点是完全无损保存。运用不多，基本不应用于网络，但是Windows系统的标准图片格式，（几乎）所有Windows看图/编辑软件应该都支持。
        # tiff格式:但是压缩比很低，所以和bmp并差不了多少，同样保真度也很高。现在基本上是看不到了，比bmp还少。
        # jpg格式:它用有损压缩方式去除冗余的图像和彩色数据，在获得极高的压缩率的同时能展现十分丰富生动的图像，即可以用较少的磁盘空间得到较好的图片质量（但稍逊色于png）。
        # png格式:可以做到几乎无损的压缩，而且压缩比挺高的，大概是Bmp的10几或几十分之一吧，质量很高，支持透明，90年代出现，至今用途广泛，常用于Internet，和jpg和gif都是网络图片格式。


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Mywindow()
    window.show()
    sys.exit(app.exec_())

