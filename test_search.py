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
    y.append(pca_point_cloud[i][1])
  plt.scatter(x, y)
  plt.show()


# 查询接口中每行代码执行的时间
def func_line_time(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        func_return = f(*args, **kwargs)
        lp = LineProfiler()
        lp_wrap = lp(f)
        lp_wrap(*args, **kwargs)
        lp.print_stats()
        return func_return

    return decorator


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

def Rationality(input,pre):
  temp = 0
  i = 0
  length = len(input)
  while(i<length):
    if(abs(input[i]-pre[i])<=0.08):
      temp += 1
    i += 1

  return np.double(temp)/length


def optimize(x,y,z1,p1,ration):
  xx = x
  yy = y
  temp_x = x
  temp_y = y
  best_p = z1
  best_z = p1
  temp_ration = ration
  best_ration = ration
  i = 0
  k = 0
  while i < 1:
    k += 1
    temp_ration =best_ration
    for num in range(0,len(xx),10):
      if (num + 8 < len(xx)):
        x_new = np.delete(xx, [num, num + 1, num + 2, num + 3, num + 4,num+5,num+6,num+7,num+8,num+9])#删除num开始的五个点
        y_new = np.delete(yy, [num, num + 1, num + 2, num + 3, num + 4,num+5,num+6,num+7,num+8,num+9])
        # x_new = np.delete(xx, [num, num + 1, num + 2, num + 3, num + 4])  # 删除num开始的五个点
        # y_new = np.delete(yy, [num, num + 1, num + 2, num + 3, num + 4])
        z1_new = np.polyfit(x_new, y_new, 3)
        p1_new = np.poly1d(z1_new)  # 返回值为多项式的表达式，也就是函数式子
        y_prednew = p1_new(x)  # 根据函数的多项式表达式，求解 y
        newrationality = Rationality(y, y_prednew)
        if newrationality> best_ration:
          temp_x = x_new
          temp_y = y_new
          best_p = p1_new
          best_z = z1_new
          best_ration = newrationality
    xx=temp_x
    yy=temp_y
    if temp_ration == best_ration:
      i = 1
      print(best_ration)
  print("k=",k)

  return best_p,best_z

def roulette(input,pre,):
  temp = 0
  i = 0
  length = len(input)
  while (i < length):
    if (abs(input[i] - pre[i]) <= 0.03):
      temp += 1
    i += 1

  return np.double(temp) / length




# @func_line_time
# 定义一个测试函数
def display():

  afile='fanfan1'
  #设fan为我们的正确的方向
  # txt_path= '../txtcouldpoint/Finalzhengfan5.txt'#负的h1, h2, h3,但是后面的所有的都是对的
  # txt_path= '../txtcouldpoint/Finalzhengzheng1.txt'#正的h1, h2, h3,但是后面的所有的都是反的,所以这个要旋转180度
  txt_path= 'txtcouldpoint/Final{}.txt'.format(afile)

  fp = open('test/FinalOutPut/{}/20.txt'.format(afile), 'w')
  f3 = open('test/FinalOutPut/{}/30.txt'.format(afile), 'a+')
  f4 = open('test/FinalOutPut/{}/40.txt'.format(afile), 'a+')
  save_path = "test/FinalOutPut/{}/".format(afile)+"Filter_{}.png"

  np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

  start_time = time.time()
  # 通过numpy读取txt点云
  pcd_1 = np.genfromtxt(txt_path, delimiter=",")
  pcd = o3d.geometry.PointCloud()
  pcd_50percent = o3d.geometry.PointCloud()

  # 加载点坐标
  pcd.points = o3d.utility.Vector3dVector(pcd_1)
  pcd_50percent.points=pcd.points[(pcd_1.shape[0]//4):((pcd_1.shape[0]//4)*3)]
  end_time = time.time()
  print(end_time - start_time)

  pcd = pcd.translate(-pcd_50percent.get_center(), relative=True)  #平移
  pcd_50percent = pcd_50percent.translate(-pcd_50percent.get_center(), relative=True)
  firstime=time.time()
  w, v = xyz1.PCA(pcd_50percent.points)  # PCA方法得到对应的特征值和特征向量
  second_time = time.time()
  print('firstime',second_time-firstime)
  point_cloud_vector = v[:, 0]  # 点云主方向对应的向量为最大特征值对应的特征向量
  print('the main orientation of this pointcloud is: ', point_cloud_vector)
  print('v',v)
  if(v[0][0]<0):
    v[:,0]=-v[:,0]
    v[:,1]=-v[:,1]
  # if(v[0][1]>0):
  #   pre_v=1
  # else:
  #   pre_v=-1
  # print('pre_v',pre_v)
  print('v', v)




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

  mesh_1 = o3d.geometry.TriangleMesh.create_coordinate_frame()
  axis = o3d.geometry.TriangleMesh.create_coordinate_frame().rotate(v, center=(0, 0, 0))
  mesh_1.scale(20, center=(0, 0, 0))
  axis.scale(20, center=(0, 0, 0))
  # pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pcd.points))
  # 可视化
  # o3d.visualization.draw_geometries([pc_view, axis, mesh_1], point_show_normal=True)
  # o3d.visualization.draw_geometries([pc_view, mesh_1], point_show_normal=True)

  # 转化xy轴
  h1, h2, h3 = Router(v)
  # print(h1, h2, h3)
  # if ( h1 > 1 ):
  #   h1=np.pi-h1
  print(h1,h2,h3)
  print(v[1][0])
  if(v[1][0]>0):
    R1 = pcd.get_rotation_matrix_from_xyz((0, 0, -h1))
  else:
    R1 = pcd.get_rotation_matrix_from_xyz((0, 0, h1))
  # R2= pcd.get_rotation_matrix_from_xyz((0, 0,np.pi))
  # pcd.rotate(R2,center=(0,0,0))        # 旋转
  pcd.rotate(R1,center=(0,0,0))        # 旋转
  mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
  mesh.scale(20, center=(0,0,0))
  # pc_view_1 = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pcd.points))
  # # 可视化
  # o3d.visualization.draw_geometries([pc_view_1,pc_view_1, mesh], point_show_normal=True)


  # 计算要切割的值
  P2 = np.array([0, 0, 0])  # xyz
  # a, b, c, d = plane_param(point_cloud_vector,P2)
  # a, b, c, d = plane_param(v[:, 1], P2)#因为已经旋转好了，所以不用v了，直接用原始的就行了
  a, b, c, d = plane_param([0,1,0], P2)#因为已经旋转好了，所以不用v了，直接用原始的就行了
  point_size = point.shape[0]
  idx = []
  # 3.设置切片厚度阈值，此值为切片厚度的一半
  Delta = 0.05


  # 4.循环迭代查找满足切片的点
  for i in range(point_size):
    Wr = a * point[i][0] + b * point[i][1] + c * point[i][2] + d - Delta
    Wl = a * point[i][0] + b * point[i][1] + c * point[i][2] + d + Delta
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

  axis = o3d.geometry.TriangleMesh.create_coordinate_frame().rotate(v, center=(0, 0, 0))
  mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
  # pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(slicing_cloud.points))
  # # 可视化
  # o3d.visualization.draw_geometries([pc_view, mesh], point_show_normal=True)


  poi = np.asarray(slicing_cloud.points)  #转换数组
  Point_Show(poi)
  poi_x = poi[:, 0]     #切片  第一列
  pre_sort_x = sorted(enumerate(poi_x), key=lambda poi_x: poi_x[1])     #以第二个值X[1]进行排序
  sorted_poi = np.zeros((poi.shape))
  # print(len(poi_x))     #图像上点的数量
  for i in range(len(poi_x)):
    sorted_poi[i] = poi[pre_sort_x[i][0]] #赋值改点的序号
  sorrrayiex=0
  sorrrayiey=len(sorted_poi-1)
  for i in range(len(sorted_poi)):
    #print(sorted_poi[i][0])
    now_length=sorted_poi[i][0]-sorted_poi[0][0]
    if now_length >=sorted_poi[-1][0]-sorted_poi[0][0]-25 and sorrrayiex==0:    #最后2.5cm
      sorrrayiex = i    #标志位
    # if sorted_poi[i][0]-sorted_poi[0][0]>120:
    if now_length > sorted_poi[-1][0]-sorted_poi[0][0]-5:  #去掉最后5mm，减少误差
      sorrrayiey = i    #标志位
      break



  #print(sorrrayiex,len(sorted_poi))

  x = sorted_poi[:, 0]
  y = sorted_poi[:, 2]
  x = x - x[0]
  y = y - y[0]


  a=time.time()


  plt.plot(x[sorrrayiex:sorrrayiey], y[sorrrayiex:sorrrayiey], '*', label='original values')
  akb=signal.argrelmin(y[sorrrayiex:sorrrayiey], order=15)    #局部相对最小
  print("akb", akb[0][0])
  xmin=x[akb[0][0]+sorrrayiex]
  ymin=y[akb[0][0]+sorrrayiex]
  for i in akb[0]:
    # print(i)
    if (y[i+sorrrayiex]<ymin):
      ymin=y[i+sorrrayiex]
      xmin=x[i+sorrrayiex]

  plt.plot(xmin,ymin, '+',markersize=20)  # 极小值点
  plt.title('')
  plt.xlabel('')
  plt.ylabel('')
  plt.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))
  b=time.time()
  print('time',b-a)
  plt.show()


  mask =point[:, 0] < slicing_max     #比较point第一列与slicing_max，结果保存在mask
  pcd.points = o3d.utility.Vector3dVector(point[mask])
  points2=np.asarray(pcd.points)
  mask2=points2[:, 0] > slicing_min
  pcd.points = o3d.utility.Vector3dVector(points2[mask2])
  point= np.asarray(pcd.points)

  poly3 = list()
  poly2 = list()
  poly1 = list()
  poly0 = list()

  tank1=1   #每次切间隔距离（除以10为真实距离单位：mm）
  astart = time.time()
  while (slicing_min + 1+tank1*2/10 < slicing_max - 0.1):
    if tank1==7:

      tank=slicing_min + 1+tank1*2/10      #tank切的位置
      P2 = np.array([tank, 0, 0])  # xyz
      # a, b, c, d = plane_param(point_cloud_vector,P2)
      # a, b, c, d = xyz1.plane_param(v[:, 0], P2)#因为已经旋转好了，所以直接使用原始值就行了.
      a, b, c, d = xyz1.plane_param([-1,0,0], P2)#因为已经旋转好了，所以直接使用原始值就行了.
      point_size = point.shape[0]
      idx = []
      # 3.设置切片厚度阈值，此值为切片厚度的一半
      Delta = 0.2
      # 4.循环迭代查找满足切片的点
      for i in range(point_size):
        Wr = a * point[i][0] + b * point[i][1] + c * point[i][2] + d - Delta
        Wl = a * point[i][0] + b * point[i][1] + c * point[i][2] + d + Delta
        if ((Wr < 0)and(Wl>0)) or ((Wr>0) and (Wl <0)):
          idx.append(i)
      # 5.提取切片点云
      slicing_cloud = (pcd.select_by_index(idx))
      slicing_points = np.asarray(slicing_cloud.points)

      # axis = o3d.geometry.TriangleMesh.create_coordinate_frame().rotate(v, center=(0, 0, 0))
      # pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pcd.points))
      # # 可视化
      # o3d.visualization.draw_geometries([pc_view, axis], point_show_normal=True)

      project_pane = [a, b, c, d]
      points_new = xyz1.point_project_array(slicing_points, project_pane)
      pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points_new))
      # xyz1.Point_Show(points_new)

      # 转化xy轴
      h1, h2, h3 = Router(v)
      # R1 = pcd.get_rotation_matrix_from_xyz((h1, h2, h3))
      # R2 = pcd.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))
      # pc_view.rotate(R1)

      # axis = o3d.geometry.TriangleMesh.create_coordinate_frame().rotate(v, center=(0, 0, 0))
      # pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc_view.points))
      # # 可视化
      # o3d.visualization.draw_geometries([pc_view, axis], point_show_normal=True)

      # pc_view.rotate(R2)

      poi = np.asarray(pc_view.points)
      # xyz1.Point_Show(poi)
      poi_x = poi[:, 1]
      pre_sort_x = sorted(enumerate(poi_x), key=lambda poi_x: poi_x[1])
      sorted_poi = np.zeros((poi.shape))
      for i in range(len(poi_x)):
        sorted_poi[i] = poi[pre_sort_x[i][0]]



      #将图像中点坐标转移到0，0,
      x = sorted_poi[:, 1]
      # print(sorted_poi,sorted_poi)
      y = sorted_poi[:, 2]
      # print(sorted_poi)
      # plt.plot(x, y, '*')
      # plt.show()
      ymax_index = np.argmax(y)
      akb1 = signal.argrelmax(y, order=40)  # 局部相对最大
      # print(num)
      # x = x - x[ymax_index]
      # y = y - y[ymax_index]
      # x = x - x[median]
      # y = y - y[median]
      # x = x - x[0]
      # y = y - y[0]



      #现在是没有对齐，就是初始的状态的曲线拟合



      if np.size(akb1)<=1:
        # x = x - x[akb1]
        # y = y - y[akb1]
        z1 = np.polyfit(x, y, 3)  # 曲线拟合，返回值为多项式的各项系数
      elif np.size(akb1)>1:
        median = int(len(poi_x) / 2)
        # x = x - x[median]
        # y = y - y[median]
        z1 = np.polyfit(x, y, 4)  # 曲线拟合，返回值为多项式的各项系数

      p1 = np.poly1d(z1)  # 返回值为多项式的表达式，也就是函数式子
      y_pred = p1(x)  # 根据函数的多项式表达式，求解 y
      # print(y_pred-y)
      dd = abs(y_pred-y)
      cc = np.mean(dd)
      print(cc)
      rationality = Rationality(y,y_pred)
      print("--------")
      print(rationality)

      if(abs(cc)>0.01):
        p1, z1 = optimize(x, y, z1, p1, rationality)
        y_pred = p1(x)
        # for num in range[0,len(cc),5]:
          # if(num+4<=len(cc)):
          #   x_new = np.delete(x,[num,num+1,num+2,num+3,num+4])
          #   y_new = np.delete(y,[num,num+1,num+2,num+3,num+4])
          #   z1_new = np.polyfit(x_new, y_new, 3)
          #   p1_new = np.poly1d(z1_new)  # 返回值为多项式的表达式，也就是函数式子
          #   y_prednew = p1_new(x)  # 根据函数的多项式表达式，求解 y
          #   # print(y_pred-y)
          #   # dd_new = y_prednew - y
          #   # cc_new = np.mean(dd_new)
          #   newrationality = Rationality(y,y_prednew)
          #
          # print("1")




      # print(cc)
      # po = 3
      # while(abs(cc)>1.5e-16):
      #   po += 1
      #   zz = np.polyfit(x,y,po)
      #   p1 = np.poly1d(zz)
      #   y_pred=p1(x)
      #   dd = y_pred - y
      #   cc = np.mean(dd)
      # print("start add :")
      # print(po)


      z2 = np.asarray(z1)

      plt.plot(x[akb1], y[akb1], '+', markersize=20)
      plt.plot(x, y, '*', label='original values')



      plt.plot(x, y_pred, label='fit values')
      plt.title('')
      plt.xlabel('')
      plt.ylabel('')
      plt.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))
      # plt.show()
      # saveName=tank1
      if (tank-xmin-slicing_min-1 <=2 and tank-xmin-slicing_min-1>=-2):#因为拐点的范围比较大，比0.2mm要大得多，所以如果这里用0.2mm的话，那么这边边上一片邻域都是和它接近一模一样的拐点。
        save_plt="test/FinalOutPut/{}/".format(afile)
        plt.savefig(save_plt+"Filter_One_{}.png".format(tank1))
        plt.clf()
      else:
        if np.size(akb1)<=1:
          #保存3次拟合的各项系数
          data_list = []
          data_list.append(z2[0])
          data_list.append(z2[1])
          data_list.append(z2[2])
          # data_list.append(z2[3])
          list1 = json.dumps(data_list)
          f3.write(list1 + "\n")

        else:
          data_list = []
          data_list.append(z2[0])
          data_list.append(z2[1])
          data_list.append(z2[2])
          data_list.append(z2[3])
          data_list.append(z2[4])
          list1 = json.dumps(data_list)
          f4.write(list1 + "\n")

        plt.savefig(save_path.format(tank1))
        plt.clf()
      # print('-----------------',tank1,'--------------------',file=fp)
      # print(p1,file=fp)
      # print(z2[0])
      # poly3 = poly3.append(z2[0])

      poly3.append(z2[0])
      poly2.append(z2[1])
      poly1.append(z2[2])
      # poly0.append(z2[3])

      print(z2, file=fp)
      tank1 += 1
      print(tank1)
    else:
      tank1 += 1


  bstart = time.time()
  print(bstart - astart)
  fp.close()
  f3.close()
  f4.close()


if __name__ == "__main__":

  display()
  # x = [0,1,7,9,10]
  # y = [0,1,2,4,5]
  # z = np.delete(x,[1,2,3])
  # # x.pop(2,4)
  #
  # # c=Rationality(x,y)
  # print(x)
  # print(z)
