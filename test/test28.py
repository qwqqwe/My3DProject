'''

将图像按照不同方式进行对齐
测试，尝试用拐点对齐
'''
import json
import line_profiler
import time
import xyz1
import cv2
import scipy.linalg as linalg
import math
import scipy
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from mayavi import mlab
from pyinstrument import Profiler
from line_profiler import LineProfiler
import heartrate
import random
from scipy import signal
from kneed import KneeLocator
from line_profiler import LineProfiler
from functools import wraps


def display_inlier_outlier(cloud, ind):
  inlier_cloud = cloud.select_by_index(ind)
  outlier_cloud = cloud.select_by_index(ind, invert=True)

  print("Showing outliers (red) and inliers (gray): ")
  outlier_cloud.paint_uniform_color([1, 0, 0])
  inlier_cloud.paint_uniform_color([0, 1, 0])
  o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
def plane_param(point_cloud_vector, point):
  """
  不共线的三个点确定一个平面
  :return: 平面方程系数:a,b,c,d
  """
  # n1 = point_cloud_vector / np.linalg.norm(point_cloud_vector)  # 单位法向量
  n1 = point_cloud_vector
  # print(n1)
  A = n1[0]
  B = n1[1]
  C = n1[2]
  D = -A * point[0] - B * point[1] - C * point[2]
  return A, B, C, D


def Point_Show(pca_point_cloud):
  x = []
  y = []
  pca_point_cloud = np.asarray(pca_point_cloud)
  for i in range(len(pca_point_cloud)):
    x.append(pca_point_cloud[i][0])
    y.append(pca_point_cloud[i][2])
  plt.scatter(x, y)
  plt.show()
def Router(v):
  # 求向量V与标准xyz坐标的角度
  x1 = np.array((1, 0, 0))
  y1 = v[:, 0]
  x2 = np.array((0, 1, 0))
  y2 = v[:, 1]
  x3 = np.array((0, 0, 1))
  y3 = v[:, 2]
  l_x1 = np.sqrt(x1.dot(x1))
  l_y1 = np.sqrt(y1.dot(y1))
  dian1 = x1.dot(y1)  # x1点积y1
  cos_1 = dian1 / (l_x1 * l_y1)
  angle_hu1 = np.arccos(cos_1)
  l_x2 = np.sqrt(x2.dot(x2))
  l_y2 = np.sqrt(y2.dot(y2))
  dian2 = x2.dot(y2)
  cos_2 = dian2 / (l_x2 * l_y2)
  angle_hu2 = np.arccos(cos_2)
  l_x3 = np.sqrt(x3.dot(x3))
  l_y3 = np.sqrt(y3.dot(y3))
  dian3 = x3.dot(y3)
  cos_3 = dian3 / (l_x3 * l_y3)
  angle_hu3 = np.arccos(cos_3)
  return angle_hu1, angle_hu2, angle_hu3

# @func_line_time
# 定义一个测试函数

def display():

  txt_path= '../txtcouldpoint/Others/qiepian.txt'
  np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
  qiepian=np.loadtxt(txt_path, delimiter=",")
  plt.scatter(qiepian[:,0], qiepian[:,1])
  plt.show()

if __name__ == "__main__":

  display()

