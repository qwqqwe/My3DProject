from OpenGL.GL import *
from OpenGL.GLUT import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow,QOpenGLWidget

class MyOpenglWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def initializeGL(self):
        glClearColor(0, 0, 0, 1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHTING)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)


        # pass

    def paintGL(self):
        glBegin(GL_LINES)  # 开始绘制线段（世界坐标系）

        # 以红色绘制x轴
        glColor4f(1.0, 0.0, 0.0, 1.0)  # 设置当前颜色为红色不透明
        glVertex3f(-0.8, 0.0, 0.0)  # 设置x轴顶点（x轴负方向）
        glVertex3f(0.8, 0.0, 0.0)  # 设置x轴顶点（x轴正方向）

        # 以绿色绘制y轴
        glColor4f(0.0, 1.0, 0.0, 1.0)  # 设置当前颜色为绿色不透明
        glVertex3f(0.0, -0.8, 0.0)  # 设置y轴顶点（y轴负方向）
        glVertex3f(0.0, 0.8, 0.0)  # 设置y轴顶点（y轴正方向）

        # 以蓝色绘制z轴
        glColor4f(0.0, 0.0, 1.0, 1.0)  # 设置当前颜色为蓝色不透明
        glVertex3f(0.0, 0.0, -0.8)  # 设置z轴顶点（z轴负方向）
        glVertex3f(0.0, 0.0, 0.8)  # 设置z轴顶点（z轴正方向）

        glEnd()  # 结束绘制线段
        # pass

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # gluPerspective(45, w / h, 0.01, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)


        # pass

if __name__=="main":
    app = QApplication(sys.argv)
    # mainWindow = QMainWindow()
    ui = MyOpenglWidget()
    # ui.setupUi(mainWindow)
    ui.show()
    sys.exit(app.exec_())
