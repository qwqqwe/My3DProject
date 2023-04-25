import sys
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import pyqtSignal,QObject

import re
import numpy as np
import open3d as o3d
from ctypes import *

import configparser
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from matplotlib import pyplot as plt
from scipy import signal
from line_profiler import LineProfiler
from functools import wraps

def point_project_array(points, para):
  # 这是投影的函数，投影到一个平面上
  para = np.array(para)
  d = para[0] ** 2 + para[1] ** 2 + para[2] ** 2
  t = -(np.matmul(points[:, :3], para[:3].T) + para[3]) / d
  points = np.matmul(t[:, np.newaxis], para[np.newaxis, :3]) + points[:, :3]
  return points

def PCA(data, correlation=False, sort=True):
  # normalize 归一化
  mean_data = np.mean(data, axis=0)
  normal_data = data - mean_data
  # 计算对称的协方差矩阵
  H = np.dot(normal_data.T, normal_data)
  # SVD奇异值分解，得到H矩阵的特征值和特征向量
  eigenvectors, eigenvalues, _ = np.linalg.svd(H)

  if sort:
    sort = eigenvalues.argsort()[::-1]  # 对特征向量进行排序，从大到小，返回索引值
    eigenvalues = eigenvalues[sort]
    eigenvectors = eigenvectors[:, sort]

  return eigenvalues, eigenvectors
def display_inlier_outlier(cloud, ind):
  inlier_cloud = cloud.select_by_index(ind)
  outlier_cloud = cloud.select_by_index(ind, invert=True)

  print("Showing outliers (red) and inliers (gray): ")
  outlier_cloud.paint_uniform_color([1, 0, 0])
  inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
  o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def Rationality(input,pre,findbest):
  temp = 0
  i = 0
  length = len(input)
  while(i<length):
    if(abs(input[i]-pre[i])<=findbest):
      temp += 1
    i += 1

  return np.double(temp)/length


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

def function(x):
  if abs(x)<=0.634:
    asd = -0.4556 * x * x - 0.0392
  else:
    asd = -0.55248 * x * x
  return asd

def display2():
# def display2(pcd_1):
  # if len(pcd_1)==0:
  #   return np.zeros([1,4]),["扫描数据为空"]

  afile='fanzheng5'
  txt_path= '../../txtcouldpoint/Final{}.txt'.format(afile)

  defect_meassage=[]
  # 通过numpy读取txt点云
  pcd_1=np.loadtxt(txt_path, delimiter=",")
  pcd = o3d.geometry.PointCloud()
  pcd_50percent = o3d.geometry.PointCloud()

  # 加载点坐标
  pcd.points = o3d.utility.Vector3dVector(pcd_1)
  pcd_50percent.points=pcd.points[(pcd_1.shape[0]//4):((pcd_1.shape[0]//4)*3)]

  pcd = pcd.translate(-pcd_50percent.get_center(), relative=True)  #平移
  pcd_50percent = pcd_50percent.translate(-pcd_50percent.get_center(), relative=True)
  w, v = PCA(pcd_50percent.points)  # PCA方法得到对应的特征值和特征向量

  point_cloud_vector = v[:, 0]  # 点云主方向对应的向量为最大特征值对应的特征向量
  print('the main orientation of this pointcloud is: ', point_cloud_vector)
  if(v[0][0]<0):
    v[:,0]=-v[:,0]
    v[:,1]=-v[:,1]

  pcd = pcd.uniform_down_sample(50) #均匀下采样，50个点取一个点

  # ------------------------- 统计滤波 --------------------------
  print("->正在进行统计滤波...")
  astart = time.time()
  num_neighbors = 20  # K邻域点的个数
  std_ratio = 2.0  # 标准差乘数
  # 执行统计滤波，返回滤波后的点云sor_pcd和对应的索引ind  去除离群点
  sor_pcd, ind = pcd.remove_statistical_outlier(num_neighbors, std_ratio)   #目标点相邻点个数，偏离标准差的倍速，返回元组，列表
  #在一个点周围选择若干个点，计算它们距离的统计参数，如果某个点偏离平均值超过stdio_ratio倍的方差则认为是离群点
  #remove_radius_outlier（points,radius）目标点周围指定半径内统计点的数量，如果点的数量小于某一阈值则认为目标点是离群点并进行删除
  bstartime = time.time()
  print("统计滤波", bstartime - astart)
  # 可视化统计滤波后的点云和噪声点云
  # display_inlier_outlier(pcd, ind)

  pcd = sor_pcd
  points = pcd.points
  point = np.asarray(points)

  # 转化xy轴
  h1, h2, h3 = Router(v)
  print(h1,h2,h3)
  print(v[1][0])
  if(v[1][0]>0):
    R1 = pcd.get_rotation_matrix_from_xyz((0, 0, -h1))
  else:
    R1 = pcd.get_rotation_matrix_from_xyz((0, 0, h1))
  R2= pcd.get_rotation_matrix_from_xyz((0, 0,np.pi))#如果后缀是zheng的话，需要把这个启用
  pcd.rotate(R2,center=(0,0,0))        # 旋转
  pcd.rotate(R1,center=(0,0,0))        # 旋转

  # 计算要切割的值
  point_size = point.shape[0]
  idx = []
  # 3.设置切片厚度阈值，此值为切片厚度的一半
  Delta = 0.05

  # 4.循环迭代查找满足切片的点
  for i in range(point_size):
    Wr = point[i][1] - Delta
    Wl = point[i][1] + Delta
    if ((Wr < 0)and(Wl>0)) or ((Wr>0) and (Wl <0)):
      idx.append(i)
  # 5.提取切片点云
  slicing_cloud = (pcd.select_by_index(idx))
  slicing_points = np.asarray(slicing_cloud.points)
  slicing_min = slicing_points[0][0]
  slicing_max = slicing_points[-1][0]
  if (slicing_min > slicing_max):
    slicing_min = slicing_points[-1][0]
    slicing_max = slicing_points[0][0]

  poi = np.asarray(slicing_cloud.points)  #转换数组
  poi_x = poi[:, 0]     #切片  第一列
  pre_sort_x = sorted(enumerate(poi_x), key=lambda poi_x: poi_x[1])     #以第二个值X[1]进行排序
  sorted_poi = np.zeros((poi.shape))
  for i in range(len(poi_x)):
    sorted_poi[i] = poi[pre_sort_x[i][0]] #赋值改点的序号
  sorrrayiex=0
  sorrrayiey=len(sorted_poi-1)
  for i in range(len(sorted_poi)):
    now_length=sorted_poi[i][0]-sorted_poi[0][0]
    if now_length >=sorted_poi[-1][0]-sorted_poi[0][0]-25 and sorrrayiex==0:    #最后2.5cm
      sorrrayiex = i    #标志位
    if now_length > sorted_poi[-1][0]-sorted_poi[0][0]-5:  #去掉最后5mm，减少误差
      sorrrayiey = i    #标志位
      break

  x = sorted_poi[:, 0]
  y = sorted_poi[:, 2]
  x = x - x[0]
  y = y - y[0]

  akb=signal.argrelmin(y[sorrrayiex:sorrrayiey], order=15)    #局部相对最小
  xmin=x[akb[0][0]+sorrrayiex]
  ymin=y[akb[0][0]+sorrrayiex]
  for i in akb[0]:
    if (y[i+sorrrayiex]<ymin):
      ymin=y[i+sorrrayiex]
      xmin=x[i+sorrrayiex]

  mask =point[:, 0] < slicing_max     #比较point第一列与slicing_max，结果保存在mask
  pcd.points = o3d.utility.Vector3dVector(point[mask])
  points2=np.asarray(pcd.points)
  mask2=points2[:, 0] > slicing_min
  pcd.points = o3d.utility.Vector3dVector(points2[mask2])
  point= np.asarray(pcd.points)

  tank1=1   #每次切间隔距离（除以10为真实距离单位：mm）
  astart = time.time()
  step = 0  #记录裂缝位置
  step1=0   #记录弧坑及焊瘤位置
  type_label=1

  list_all = []

  #改进切片
  if(point[0][0]>0):
    point=point[::-1]
  tank = slicing_min + 1 + tank1 * 2 / 10  # tank切的位置
  point_size = point.shape[0]
  idx = []
  idx_list=[]
  # 3.设置切片厚度阈值，此值为切片厚度的一半
  Delta = 0.2
  now_i=0
  for i in range(point_size):
    Wr = -point[i][0] + tank - Delta
    Wl = -point[i][0] + tank + Delta
    if ((Wr < 0) and (Wl > 0)) or ((Wr > 0) and (Wl < 0)):
      idx.append(i)
      now_i=i
  idx_list.append(idx)
  idx=[]
  tank1+=1
  tank = slicing_min + 1 + tank1 * 2 / 10  # tank切的位置
  for i in range(now_i,point_size):
    if (point[i][0]<=tank+Delta):
      idx.append(i)
    else:
      idx_list.append(idx)
      idx=[]
      tank1+=1
      tank = slicing_min + 1 + tank1 * 2 / 10  # tank切的位置
      if(tank>=slicing_max-0.1):
        break
  for tank1,recent_idx in enumerate(idx_list):
    no_data=0
    # 4.循环迭代查找满足切片的点################################
    # 第二次切片 ##### ##### ##### ##### ##### ##### ##### ######  ##### ##### #####  ######  ##### ##### #####
    # 5.提取切片点云
    tank = slicing_min + 1 + tank1 * 2 / 10  # tank切的位置
    slicing_cloud = (pcd.select_by_index(recent_idx))
    slicing_points = np.asarray(slicing_cloud.points)


    project_pane = [-1, 0, 0, tank]
    points_new = point_project_array(slicing_points, project_pane)#这是投影的函数，投影到一个平面上
    pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points_new))


    poi = np.asarray(pc_view.points)
    poi_x = poi[:, 1]
    pre_sort_x = sorted(enumerate(poi_x), key=lambda poi_x: poi_x[1])
    sorted_poi = np.zeros((poi.shape))
    for i in range(len(poi_x)):
      sorted_poi[i] = poi[pre_sort_x[i][0]]

    #判断是否有空白#
    #如果数据长度小于1.75的话，判断有点遗失，数据量不够
    #TODO 将里面电脑缺陷判断进行替换，同时最好将函数放置到函数内部，方便修改阈值
    if (len(sorted_poi)==0):
      no_data = 1
      print("缺焊")
      print(tank1)
      defect_meassage.append("缺焊:"+str(tank1))
      # print(tank1, file=f5)
    elif(-0.75<(sorted_poi[-1][1]- sorted_poi[0][1])<0.75):
      no_data = 1
      print("noda")
      print(tank1)
      # print(tank1, file=f5)
      defect_meassage.append("no_data:" + str(tank1))
    else:
      for i in range(len(poi_x)-1):
        #如果两个点之间差距大于0.5的话，判断点有遗失，数据量不够
        if((sorted_poi[i+1][1]- sorted_poi[i][1])>1.5):
          no_data=1
          print("no_data")
          print(tank1)
          defect_meassage.append("no_data:" + str(tank1))
          # print(tank1,file=f5)
          break

      print(sorted_poi[i+1][1]- sorted_poi[i][1])

    if (no_data!=1):

      x_original = sorted_poi[:, 0]
      y_original = sorted_poi[:, 1]
      z_original = sorted_poi[:, 2]
      yyy = y_original[0:len(y_original) // 3]
      yyy = np.hstack((yyy, y_original[len(y_original) * 2 // 3:]))
      zzz = z_original[0:len(z_original) // 3]
      zzz = np.hstack((zzz, z_original[len(z_original) * 2 // 3:]))  # 边缘部分
      yy = y_original[len(y_original) // 6:len(y_original) *5// 6]#中间段
      zz = z_original[len(z_original) // 6:len(z_original) *5// 6]
      akb1 = signal.argrelmax(z_original, order=10)  # 局部相对最大



      if np.size(akb1)<=1:
        # z1 = np.polyfit(yy, zz, 2)  # 曲线拟合，返回值为多项式的各项系数
        zz1 = np.polyfit(yyy, zzz, 2)  # 曲线拟合，返回值为多项式的各项系数

      elif np.size(akb1)>1:
        # z1 = np.polyfit(y_original, z_original, 4)  # 曲线拟合，返回值为多项式的各项系数
        zz1 = np.polyfit(yyy, zzz, 2)  # 曲线拟合，返回值为多项式的各项系数

      #找到函数的最高点并进行平移
      max_y=zz1[1]/(-zz1[0]*2)
      ################################################################
      pp1 = np.poly1d(zz1)  # 返回值为多项式的表达式，也就是函数式子
      ################################################################

      max_z = pp1(max_y)
      y_adjusted = y_original - max_y
      z_adjusted = z_original - max_z  # 减去最高点
      z_pred = []


      for num in range(0,len(y_adjusted)):
        z_pred.append(function(y_adjusted[num]))

      if np.size(akb1)<=1:
        if step1 != 0:
          print("弧坑起始位置", tank1 - step1)
          print("弧坑结束位置", tank1)
          defect_meassage.append("弧坑起始位置:" + str(tank1 - step1))
          defect_meassage.append("弧坑结束位置:" + str(tank1))
          step1 =0

        z_adjusted=z_adjusted-np.mean(z_adjusted-z_pred)
        score=Rationality(z_adjusted,z_pred,0.05)
        # score=Rationality(z_adjusted,z_pred,0.1)


        if(score<=0.9):#得分低的，将重新进行2次拟合进行第二次判断
          z1 = np.polyfit(y_adjusted,z_adjusted,2)
          p1 = np.poly1d(z1)
          z_pred=p1(y_adjusted)
          score=Rationality(z_adjusted,z_pred,0.05)
          if(score<=0.9):
            # print("输出图像")
            step += 1
            # type_label=1
            for i in range(len(z_adjusted)):
              list_1 = []
              if (abs(z_adjusted[i] - z_pred[i]) <= 0.05):
                list_1.append(x_original[i])
                list_1.append(y_original[i])
                list_1.append(z_original[i])
                list_1.append(1)
                list_all.append(list_1)


              else:
                list_1.append(x_original[i])
                list_1.append(y_original[i])
                list_1.append(z_original[i])
                list_1.append(0.5)
                list_all.append(list_1)



          else:
            if (step != 0 or step1 !=0):#输出缺陷位置
              if step != 0:
                print("缺陷起始位置", tank1 - step)
                print("缺陷结束位置", tank1)
                defect_meassage.append("裂缝起始位置:" + str(tank1 - step))
                defect_meassage.append("裂缝结束位置:" + str(tank1))
                step = 0
              # else:
              if step1 != 0:
                print("弧坑起始位置", tank1 - step1)
                print("弧坑结束位置", tank1)
                defect_meassage.append("弧坑起始位置:" + str(tank1 - step1))
                defect_meassage.append("弧坑结束位置:" + str(tank1))


            # type_label=1
            for i in range(len(z_adjusted)):
              list_1 = []
              list_1.append(x_original[i])
              list_1.append(y_original[i])
              list_1.append(z_original[i])
              list_1.append(1)
              list_all.append(list_1)



          # for i in range(len(z_adjusted)):
          #   list_1=[]
          #   if(abs(z_adjusted[i]-z_pred[i])<=0.05):
          #     list_1.append(x_original[i])
          #     list_1.append(y_original[i])
          #     list_1.append(z_original[i])
          #     list_1.append(1)
          #     list_all.append(list_1)
          #     type_label=1
          #
          #   else:
          #     list_1.append(x_original[i])
          #     list_1.append(y_original[i])
          #     list_1.append(z_original[i])
          #     list_1.append(0.5)
          #     list_all.append(list_1)
          #     type_label=0.5


        else:
          if (step!=0 or step1 != 0):#输出缺陷位置
            if step!=0:
              print("缺陷起始位置", tank1 - step)
              print("缺陷结束位置", tank1)
              defect_meassage.append("裂缝起始位置:" + str(tank1 - step))
              defect_meassage.append("裂缝结束位置:" + str(tank1))

            # else:
            if step1 != 0:
              print("弧坑起始位置", tank1 - step1)
              print("弧坑结束位置", tank1)
              defect_meassage.append("弧坑起始位置:" + str(tank1 - step1))
              defect_meassage.append("弧坑结束位置:" + str(tank1))

          step = 0

          # type_label = 1
          for i in range(len(z_adjusted)):
            list_1=[]
            list_1.append(x_original[i])
            list_1.append(y_original[i])
            list_1.append(z_original[i])
            list_1.append(1)
            list_all.append(list_1)

        # step1 = 0


        if(tank+0.2>=slicing_max - 0.1):#焊缝末尾判断输出
          if (step!=0 or step1 !=0 ):
            if step!=0:
              print("缺陷起始位置", tank1 - step)
              print("缺陷结束位置", tank1)
              defect_meassage.append("裂缝起始位置:" + str(tank1 - step))
              defect_meassage.append("裂缝结束位置:" + str(tank1))
              step=0
            # else:
            if step1 !=0:
              print("弧坑起始位置", tank1 - step1)
              print("弧坑结束位置", tank1)
              defect_meassage.append("弧坑起始位置:" + str(tank1 - step1))
              defect_meassage.append("弧坑结束位置:" + str(tank1))

        step1 = 0


        # if (tank-xmin-slicing_min-1 <=2 and tank-xmin-slicing_min-1>=-2):#因为拐点的范围比较大，比0.2mm要大得多，所以如果这里用0.2mm的话，那么这边边上一片邻域都是和它接近一模一样的拐点。
        #   start_guai=1
        #   None
        # else:
        #   None

      else:#else判断的是波峰至少有两个，akb2是判断是否有俩个及两个以上的
        # step=0
        step1 +=1
        type_label=0
        akb2 = signal.argrelmin(z_original[akb1[0][0]:akb1[0][-1]], order=10)  # 局部相对最小
        if np.size(akb2)<=1:
          for i in range(len(z_adjusted)):
            list_1=[]
            list_1.append(x_original[i])
            list_1.append(y_original[i])
            list_1.append(z_original[i])
            if(i>=akb1[0][0] and i<=akb1[0][-1]):
              list_1.append(0)#添加的是弧坑
            else:
              list_1.append(1)
            list_all.append(list_1)
        if step !=0:
          print("缺陷起始位置", tank1 - step)
          print("缺陷结束位置", tank1)
          defect_meassage.append("裂缝起始位置:" + str(tank1 - step))
          defect_meassage.append("裂缝结束位置:" + str(tank1))
          step = 0

      print(tank1)

  result11 = np.array(list_all)
  bstart = time.time()
  print(bstart - astart)

  return result11,defect_meassage


def select_type(pcd,type):
    pd=[]
    for i in range (len(pcd)):
        if pcd[i][3]==type:
            print(pcd[i])
            pdd=[]
            pdd.append(pcd[i][0])
            pdd.append(pcd[i][1])
            pdd.append(pcd[i][2])
            # pdd.append(0)

            pd.append(pdd)
    return pd


def polygon_area(polygon):
  """
  compute polygon area
  polygon: list with shape [n, 3], n is the number of polygon points
  """
  area = 0
  q = polygon[1]
  for p in polygon:
    area += p[0] * q[1] - p[1] * q[0]
    q = p
  return abs(area) / 2.0

def compute_curvature(points, k=20):
  """
  计算点云的曲率
  Parameters
  ----------
  points : numpy.ndarray
      点云数组，形状为(n,3)，n为点的数量，每个点包含三个坐标值
  k : int
      用于计算曲率的邻居点数量
  Returns
  -------
  curvature : numpy.ndarray
      点云曲率数组，形状为(n,)
  """
  pcd_point = points[:][:, 0:3]

  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(pcd_point)
  # pcd.points = o3d.utility.Vector3dVector(points)
  # pcd = pcd.uniform_down_sample(50)
  pcd_tree = o3d.geometry.KDTreeFlann(pcd)
  curvature = []
  for i, p in enumerate(pcd_point):
    # for i, p in enumerate(points):
    [k, idx, _] = pcd_tree.search_knn_vector_3d(p, k)
    if k < 3 and points[i][3]==1:
      # curvature.append(0)
      continue
    if points[i][3] !=1:
      # cov = np.cov(points[idx].T)
      cov = np.cov(pcd_point[idx].T)
      eigvals = np.linalg.eigvalsh(cov)
      curvature.append(eigvals[0] / (eigvals[0] + eigvals[1] + eigvals[2]))

  print(curvature)
  np.savetxt("curvature1.txt", curvature)
  return curvature




if __name__ == '__main__':
    pcd,_ = display2()
    pd=[]
    print(np.size(pcd))
    # points[][]=pcd[:][:]
    # if pcd[:][3]==0:
    #     pd.append()
    #筛选数据
    # print(pcd[:][0:2])
    pcd_point=pcd[:][:,0:3]
    d=pcd[:][:,0:3]
    print(pcd)
    print(d)
    compute_curvature(pcd, k=20)


    # pcdd = o3d.geometry.PointCloud()
    #
    # # 加载点坐标
    # # pcdd.points = o3d.utility.Vector3dVector(pcd_point)
    # pcdd.points = o3d.utility.Vector3dVector(pd)
    # # ------------------------- 统计滤波 --------------------------
    # print("->正在进行统计滤波...")
    # astart = time.time()
    # num_neighbors = 10  # K邻域点的个数
    # std_ratio = 2.0  # 标准差乘数
    # # 执行统计滤波，返回滤波后的点云sor_pcd和对应的索引ind  去除离群点
    # # sor_pcd, ind = pcdd.remove_statistical_outlier(num_neighbors, std_ratio)  # 目标点相邻点个数，偏离标准差的倍速，返回元组，列表
    # # 在一个点周围选择若干个点，计算它们距离的统计参数，如果某个点偏离平均值超过stdio_ratio倍的方差则认为是离群点
    # min_numbers_points=2
    # radius = 0.45
    # sor_pcd, ind=pcdd.remove_radius_outlier(min_numbers_points,radius)
    # #目标点周围指定半径内统计点的数量，如果点的数量小于某一阈值则认为目标点是离群点并进行删除
    # bstartime = time.time()
    # print("统计滤波", bstartime - astart)
    # display_inlier_outlier(pcdd, ind)
    #
    # # import open3d as o3d
    # # import numpy as np
    # # import matplotlib.pyplot as plt
    #
    # # print("->正在加载点云... ")
    # # pcd = o3d.io.read_point_cloud("test.pcd")
    # # print(pcd)
    #
    # print("->正在DBSCAN聚类...")
    # eps = 0.8  # 同一聚类中最大点间距
    # min_points = 3  # 有效聚类的最小点数
    # labels = np.array(pcdd.cluster_dbscan(eps, min_points, print_progress=True))
    # max_label = labels.max()  # 获取聚类标签的最大值 [-1,0,1,2,...,max_label]，label = -1 为噪声，因此总聚类个数为 max_label + 1
    # print(f"point cloud has {max_label + 1} clusters")
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0  # labels = -1 的簇为噪声，以黑色显示
    # pcdd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcdd])
    # # point1 = []
    # # for i in range (np.array(pcdd.points).shape[0]):
    # #   print(i)
    # #   if labels[i]==1:
    # #     # print(pcdd[i])
    # #     # pin = pcdd.select_by_index(i)
    # #     print(pcdd.points[i])
    # #     print("-----")
    # #     point1.append(pcdd.points[i])
    # #     # print(pcdd.points[pin])
    # #
    # # area = polygon_area(point1)
    # # print(area)
    # # print("+++++++++++")
    #
    # for j in range (max_label+1):
    #   point1 = []
    #   for i in range(np.array(pcdd.points).shape[0]):
    #     # print(i)
    #     if labels[i] == j:
    #       # print(pcdd[i])
    #       # pin = pcdd.select_by_index(i)
    #       # print(pcdd.points[i])
    #       # print("-----")
    #       point1.append(pcdd.points[i])
    #   area = polygon_area(point1)
    #   print("第"+str(j)+"的缺陷的面积："+str(area))