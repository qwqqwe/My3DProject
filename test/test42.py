from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import vtk
from PyQt5 import QtCore, QtGui, QtWidgets
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from untitled import Ui_MainWindow
import open3d as o3d
import numpy as np

class Mywindow(QtWidgets.QMainWindow, Ui_MainWindow):

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
        txt_path = '../txtcouldpoint/Finalzhengzheng5.txt'
        pcd = np.loadtxt(txt_path, delimiter=",")

        poins = vtk.vtkPoints()
        for i in range(pcd.shape[0]):
            dp = pcd[i]
            poins.InsertNextPoint(dp[0], dp[1], dp[2])

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

        actor1 = vtk.vtkActor()
        actor1.SetMapper(dataMapper)

        self.ren.AddActor(actor)
        self.ren.AddActor(actor1)

        self.ren.ResetCamera()

        # self.frame.setLayout(self.formLayout)
        # self.setCentralWidget(self.frame)

        self.show()
        self.iren.Initialize()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Mywindow()
    window.show()
    sys.exit(app.exec_())

