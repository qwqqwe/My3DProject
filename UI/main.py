import sys
import Demo
from PyQt5.QtWidgets import QApplication, QMainWindow
import numpy as np
import open3d as o3d
import opengl_widget
class MainWindows(QMainWindow, Demo.Ui_MainWindow):

  def __init__(self, parent=None):
      QMainWindow.__init__(self, parent)
      self.setupUi(self)

      # self.pushButton.clicked.connect(self.viewtxt)

      self.pushButton.clicked.connect(opengl_widget.openGl_widget.paintGL)
  def viewtxt(self):
    y_threshold = 0.1
    print("1")
    txt_path = '..//txtcouldpoint//Finalzhengzheng5.txt'
    # txt_path = 'txtcouldpoint/Original/Third_146.txt'
    # txt_path = 'heidian.txt'


    # start_time = time.time()
    # 通过numpy读取txt点云
    pcd = np.loadtxt(txt_path, delimiter=",")
    print("25")
    pcd_vector = o3d.geometry.PointCloud()
    # print(pcd.shape)

    # 加载点坐标
    pcd_vector.points = o3d.utility.Vector3dVector(pcd[:, :3])

    #pcd_vector = pcd_vector.select_by_index(np.where(pcd[:, 2] <= y_threshold)[0])
    print("1")
    # end_time = time.time()
    # print(end_time-start_time)
    o3d.visualization.draw_geometries([pcd_vector])
    print("1")

# 创建槽函数 槽函数直接使用元件的名称即可




if __name__ == "__main__":
    app = QApplication(sys.argv)
    # mainWindow = QMainWindow()
    ui= MainWindows()
    # ui.setupUi(mainWindow)
    ui.show()
    sys.exit(app.exec_())