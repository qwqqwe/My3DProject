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
import pymysql

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
        self.IP_Detect_btn.clicked.connect(self.Ip_Detect)
        self.IP_Connect_btn.clicked.connect(self.Connect_To_Camera)
        self.btn_Save.clicked.connect(self.SaveConfig)
        self.DB_Btn_Signin.clicked.connect(self.DB_Sign_In)

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

        # self.ReadConfig()

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
        self.defect_Area(0)
        self.defect_Area(0.5)


        # # pd = select_type(self.pcd, 0.5)
        # pd = select_type(self.pcd, 0.5)
        #
        # pcdd = o3d.geometry.PointCloud()
        # # 加载点坐标
        # pcdd.points = o3d.utility.Vector3dVector(pd)
        # print("->正在DBSCAN聚类...")
        # # eps = 1.5  # 同一聚类中最大点间距
        # eps = 1.0  # 同一聚类中最大点间距
        #
        # min_points = 3  # 有效聚类的最小点数
        # labels = np.array(pcdd.cluster_dbscan(eps, min_points, print_progress=True))
        # max_label = labels.max()  # 获取聚类标签的最大值 [-1,0,1,2,...,max_label]，label = -1 为噪声，因此总聚类个数为 max_label + 1
        # print(f"point cloud has {max_label + 1} clusters")
        # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        # colors[labels < 0] = 0  # labels = -1 的簇为噪声，以黑色显示
        # pcdd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        # o3d.visualization.draw_geometries([pcdd])
        # for j in range(max_label + 1):
        #     point1 = []
        #     for i in range(np.array(pcdd.points).shape[0]):
        #         # print(i)
        #         if labels[i] == j:
        #             # print(pcdd[i])
        #             # pin = pcdd.select_by_index(i)
        #             # print(pcdd.points[i])
        #             # print("-----")
        #             point1.append(pcdd.points[i])
        #     area = polygon_area(point1)
        #     print("第" + str(j) + "的缺陷的面积：" + str(area))
        #     self.textBrowser.append("第" + str(j) + "的缺陷的面积：" + str(area))

    def defect_Area(self,type):
        pd = select_type(self.pcd, type)
        pcdd = o3d.geometry.PointCloud()
        # 加载点坐标
        pcdd.points = o3d.utility.Vector3dVector(pd)
        print("->正在DBSCAN聚类...")
        # eps = 1.5  # 同一聚类中最大点间距
        eps = 1.0  # 同一聚类中最大点间距
        min_points = 3  # 有效聚类的最小点数
        labels = np.array(pcdd.cluster_dbscan(eps, min_points, print_progress=True))
        max_label = labels.max()  # 获取聚类标签的最大值 [-1,0,1,2,...,max_label]，label = -1 为噪声，因此总聚类个数为 max_label + 1
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0  # labels = -1 的簇为噪声，以黑色显示
        pcdd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([pcdd])
        for j in range(max_label + 1):
            point1 = []
            for i in range(np.array(pcdd.points).shape[0]):
                # print(i)
                if labels[i] == j:
                    # print(pcdd[i])
                    # pin = pcdd.select_by_index(i)
                    # print(pcdd.points[i])
                    # print("-----")
                    point1.append(pcdd.points[i])
            area = polygon_area(point1)
            print("第" + str(j) + "个的"+str(type)+"缺陷的面积：" + str(area))
            self.textBrowser.append("第" + str(j) + "个的"+str(type)+"缺陷的面积：" + str(area))


    def Connect_To_Camera(self):
        a=self.Host_IP_Line.text()
        b=self.Camera_IP_Line.text()
        self.FeedBack_Text.setText(a)
        try:
            catch_back=Prepare_To_Catch(targe1t,a,b)
        except:
            print("Error")
            self.FeedBack_Text.setText('连接相机出现错误，请关闭程序再次尝试')
            return -1
        else:
            return catch_back
        #这里输入的是两个line里面的值
    def Ip_Detect(self):
        lista,strrr = Py_Detect_IP(targe1t)  # 探測IP返回的是IP的list,
        self.IP_Detect_Text.setText(strrr)
        self.FeedBack_Text.setText('IP探测')
        # self.IP_Detect_Text.setText('strrr')
        return lista
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

    def ReadConfig(self):
        config = configparser.ConfigParser()
        # 传入读取文件的地址，encoding文件编码格式，中文必须
        try:

            config.read('configs.ini', encoding='UTF-8')
        except:
            pass
        else:
            self.YuZhi_Spin.setValue(float(config['Traing']['YuZhi1']))
            self.Allowable_Error_SpinBox.setValue(float(config['Traing']['WuCha']))

            self.Host_IP_Line.setText(config['Connect_Cam']['Computer_IP'])
            self.Camera_IP_Line.setText(config['Connect_Cam']['Cam_IP'])

            self.DB_Line_Connet_Name.setText(config['DataBase']['Connect_Name'])
            self.DB_Line_Host.setText(config['DataBase']['Host'])
            self.DB_Line_Port.setText(config['DataBase']['Port'])
            self.DB_Line_Userid.setText(config['DataBase']['UserName'])
            self.DB_Line_Password.setText(config['DataBase']['PassWord'])

    def SaveConfig(self):
        config = configparser.ConfigParser()
        config['Traing'] = {}
        config['Traing']['YuZhi1'] =self.YuZhi_Spin.text()
        config['Traing']['WuCha'] =self.Allowable_Error_SpinBox.text()

        config['Connect_Cam']={}
        config['Connect_Cam']['Computer_IP']=self.Host_IP_Line.text()
        config['Connect_Cam']['Cam_IP']=self.Camera_IP_Line.text()

        config['DataBase']={}
        config['DataBase']['Connect_Name']=self.DB_Line_Connet_Name.text()
        config['DataBase']['Host']=self.DB_Line_Host.text()
        config['DataBase']['Port']=self.DB_Line_Port.text()
        config['DataBase']['UserName']=self.DB_Line_Userid.text()
        config['DataBase']['PassWord']=self.DB_Line_Password.text()

        self.FeedBack_Text.setText('保存成功')
        # db.close()

        with open('configs.ini', 'w') as configfile:
            config.write(configfile)

    def DB_Sign_In(self):
        try:
            str1=self.DB_Line_Connet_Name.text()
            host1 =self.DB_Line_Host.text()
            port1 =self.DB_Line_Port.text()
            user1=self.DB_Line_Userid.text()
            passwd1 = self.DB_Line_Password.text()
            self.FeedBack_Text.append(str1)
            global  db
            # db = pymysql.connect(host=host1, user=user1, passwd=passwd1, port=int(port1),database="test1")

            db = pymysql.connect(host='localhost', user='root', passwd='123456', port=3306,database='test1')
            # db = pymysql.connect(host='192.168.31.1', user='root', passwd='123456', port=3306)
            self.FeedBack_Text.append('连接成功！')
            print('连接成功！')
        except:
            self.FeedBack_Text.append('something wrong!')
            print('something wrong!')
        # 使用 cursor() 方法创建一个游标对象 cursor
        cursor = db.cursor()

        #sql插入语句
        sql ="""INSERT INTO test_image(id,image,txt)
                VALUES (2,'message','dir')
                """
        try:
            #执行sql语句
            cursor.execute(sql)
            #提交到数据库执行
            db.commit()
            self.FeedBack_Text.append('yes！')
        except:
            #发生错误则回滚
            db.rollback()

        # 使用 execute()  方法执行 SQL 查询
        cursor.execute("SELECT VERSION()")
        # 使用 fetchone() 方法获取单条数据.
        data = cursor.fetchone()
        self.FeedBack_Text.append('Database version : %s'% data)

        print("Database version : %s " % data)

        # 关闭数据库连接
        db.close()




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myshow = mywindow()
    myshow.show()
    sys.exit(app.exec_())
