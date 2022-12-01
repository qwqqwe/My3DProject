import sys
import Demo
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import pyqtSignal,QObject
import numpy as np
import open3d as o3d
from ctypes import *
from opengl_widget import *


# import ConnectToCamere




class MainWindows(QMainWindow, Demo.Ui_MainWindow):
  a=pyqtSignal()
  def __init__(self, parent=None):
      QMainWindow.__init__(self, parent)
      self.setupUi(self)

      # self.pushButton.clicked.connect(self.viewtxt)
      # self.widget1=openGl_widget
      self.pushButton.clicked.connect(self.pushButton_display_click)
      self.pushButton_PrepareToCatch.clicked.connect(self.pushButton_PrepareToCatch_display_click)
      self.pushButton_ToCatch.clicked.connect(self.pushButton_ToCatch_display_click)

  def pushButton_display_click(self):
      # self.textEdit_display.setText("你点击了按钮")
      bbb1=self.widget.ToCatch()
      self.widget.change()
      self.widget.cccc()
      self.widget.update()  # 刷新图像

  def pushButton_PrepareToCatch_display_click(self):
      # return_prepare = ConnectToCamere.Py_PrepareToCatch(targe1t)
      # print('return_prepare', return_prepare)
      self.widget.Pre_ToCatch()

  def pushButton_ToCatch_display_click(self):
      # bbb1 = ConnectToCamere.Py_Catch(targe1t)
      self.widget.ToCatch()

  def diaoyong(self):
      # openGl_widget.set_remark()
      self.a.emit()


# 创建槽函数 槽函数直接使用元件的名称即可




if __name__ == "__main__":
    app = QApplication(sys.argv)
    # mainWindow = QMainWindow()
    ui= MainWindows()
    # ui.setupUi(mainWindow)
    ui.show()
    sys.exit(app.exec_())