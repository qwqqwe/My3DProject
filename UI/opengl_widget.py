from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets, QtOpenGL

txt_path = '..//txtcouldpoint//Finalzhengzheng5.txt'
pcd = np.loadtxt(txt_path, delimiter=",")

class openGl_widget(QtWidgets.QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # 这个三个是虚函数, 需要重写
        # paintGL
        # initializeGL
        # resizeGL

    # 启动时会先调用 initializeGL, 再调用 resizeGL , 最后调用两次 paintGL
    # 出现窗口覆盖等情况时, 会自动调用 paintGL
    # 调用过程参考 https://segmentfault.com/a/1190000002403921
    # 绘图之前的设置
    def initializeGL(self):
        glClearColor(0, 0, 0, 1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHTING)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)

	# 绘图函数
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


        glBegin(GL_POINTS)

        c = pcd.shape[0]

        # glColor3f(1.0, 0.0, 0.0)
        for i in range(0, c):
            x = (pcd[i][0] - 70) / 70
            y = pcd[i][1] / 70
            z = pcd[i][2] / 3
            glColor3f(z, 0.0, 0.0)
            glVertex3f(x, y, z)
        glEnd()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(20, w / h, 0.01, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)

