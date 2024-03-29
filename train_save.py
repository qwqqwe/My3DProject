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
# import heartrate
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

def Rationality(input,pre,findbest):
  temp = 0
  i = 0
  length = len(input)
  while(i<length):
    if(abs(input[i]-pre[i])<=findbest):
      temp += 1
    i += 1

  return np.double(temp)/length

def function(x):
  if abs(x)<=0.634:
    asd = -0.4556 * x * x - 0.0392
  else:
    asd = -0.55248 * x * x
  return asd



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

# @func_line_time
# 定义一个测试函数

def train_coefficient():

  afile='fanzheng1'
  #设fan为我们的正确的方向
  txt_path= 'txtcouldpoint/Final{}.txt'.format(afile)

  # fp = open('test/FinalOutPut/{}/20.txt'.format(afile), 'w')
  # f3 = open('test/FinalOutPut/{}/30.txt'.format(afile), 'w')
  # f4 = open('test/FinalOutPut/{}/40.txt'.format(afile), 'w')
  # f5 = open('test/FinalOutPut/{}/50none.txt'.format(afile), 'w')
  # f6 = open('test/FinalOutPut/{}/60.txt'.format(afile), 'w')
  # f11 = open('test/FinalOutPut/{}/1_normal_x2_green_mid.txt'.format(afile), 'a+')
  # f22 = open('test/FinalOutPut/{}/2_normal_x2_red_side.txt'.format(afile), 'a+')
  # f33 = open('test/FinalOutPut/{}/3_guai_x2_green_mid.txt'.format(afile), 'a+')
  # f44 = open('test/FinalOutPut/{}/4_guai_x2_red_side.txt'.format(afile), 'a+')
  # f55 = open('test/FinalOutPut/{}/5_afterguai_x2_green_mid.txt'.format(afile), 'a+')
  # f66 = open('test/FinalOutPut/{}/6_afterguai_x2_red_side.txt'.format(afile), 'a+')

  # save_path = "test/FinalOutPut/{}/".format(afile)+"Filter_{}.png"

  np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

  start_time = time.time()
  # 通过numpy读取txt点云
  # pcd_1 = np.genfromtxt(txt_path, delimiter=",")
  pcd_1=np.loadtxt(txt_path, delimiter=",")
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
  pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pcd.points))
  # 可视化
  # o3d.visualization.draw_geometries([pc_view, axis, mesh_1], point_show_normal=True)
  # o3d.visualization.draw_geometries([pc_view, mesh_1], point_show_normal=True)

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
  mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
  mesh.scale(20, center=(0,0,0))
  pc_view_1 = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pcd.points))
  # 可视化
  # o3d.visualization.draw_geometries([pc_view_1,pc_view_1, mesh], point_show_normal=True)


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

  # axis = o3d.geometry.TriangleMesh.create_coordinate_frame().rotate(v, center=(0, 0, 0))
  # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
  # pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(slicing_cloud.points))
  # # 可视化
  # o3d.visualization.draw_geometries([pc_view, mesh], point_show_normal=True)


  poi = np.asarray(slicing_cloud.points)  #转换数组
  # Point_Show(poi)
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


  a=time.time()


  plt.plot(x[sorrrayiex:sorrrayiey], y[sorrrayiex:sorrrayiey], '*', label='original values')
  akb=signal.argrelmin(y[sorrrayiex:sorrrayiey], order=15)    #局部相对最小
  print("akb", akb[0][0])
  xmin=x[akb[0][0]+sorrrayiex]
  ymin=y[akb[0][0]+sorrrayiex]
  for i in akb[0]:
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
  # plt.show()


  mask =point[:, 0] < slicing_max     #比较point第一列与slicing_max，结果保存在mask
  pcd.points = o3d.utility.Vector3dVector(point[mask])
  points2=np.asarray(pcd.points)
  mask2=points2[:, 0] > slicing_min
  pcd.points = o3d.utility.Vector3dVector(points2[mask2])
  point= np.asarray(pcd.points)

  # poly3 = list()
  # poly2 = list()
  # poly1 = list()
  # poly0 = list()

  start_guai=0#拐点的判断
  tank1=1   #每次切间隔距离（除以10为真实距离单位：mm）
  astart = time.time()
  step = 0
  indx = 0
  list_all = []
  middle_fitting = []
  side_fitting = []


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
    points_new = xyz1.point_project_array(slicing_points, project_pane)#这是投影的函数，投影到一个平面上
    pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points_new))
    # o3d.visualization.draw_geometries([pc_view], point_show_normal=True)

    poi = np.asarray(pc_view.points)
    poi_x = poi[:, 1]
    pre_sort_x = sorted(enumerate(poi_x), key=lambda poi_x: poi_x[1])
    sorted_poi = np.zeros((poi.shape))
    for i in range(len(poi_x)):
      sorted_poi[i] = poi[pre_sort_x[i][0]]

    #判断是否有空白#
    #如果数据长度小于1.75的话，判断有点遗失，数据量不够
    if (len(sorted_poi)==0):
      no_data = 1
      print("no_data")
      print(tank1)
      # print(tank1, file=f5)
    elif(-1.75<(sorted_poi[-1][1]- sorted_poi[0][1])<1.75):
      no_data = 1
      print("no_data")
      print(tank1)
      # print(tank1, file=f5)
    else:
      for i in range(len(poi_x)-1):
        #如果两个点之间差距大于0.5的话，判断点有遗失，数据量不够
        if((sorted_poi[i+1][1]- sorted_poi[i][1])>0.5):
          no_data=1
          print("no_data")
          print(tank1)
          # print(tank1,file=f5)
          break

      print(sorted_poi[i+1][1]- sorted_poi[i][1])
    if (no_data!=1):

      #将图像中点坐标转移到0，0,

      # x = sorted_poi[:, 1]
      # y = sorted_poi[:, 2]
      # xxx=x[0:len(x)//3]
      # xxx=np.hstack((xxx, x[len(x)*2//3:]))
      # yyy=y[0:len(y)//3]
      # yyy=np.hstack((yyy, y[len(y)*2//3:]))
      # xx = x[len(x) // 6:len(x) *5// 6]
      # yy = y[len(y) // 6:len(y) *5// 6]
      # akb1 = signal.argrelmax(y, order=40)  # 局部相对最大

      x_original = sorted_poi[:, 0]
      y_original = sorted_poi[:, 1]
      z_original = sorted_poi[:, 2]
      yyy = y_original[0:len(y_original) // 3]
      yyy = np.hstack((yyy, y_original[len(y_original) * 2 // 3:]))
      zzz = z_original[0:len(z_original) // 3]
      zzz = np.hstack((zzz, z_original[len(z_original) * 2 // 3:]))  # 边缘部分
      yy = y_original[len(y_original) // 6:len(y_original) *5// 6]#中间段
      zz = z_original[len(z_original) // 6:len(z_original) *5// 6]
      akb1 = signal.argrelmax(z_original, order=40)  # 局部相对最大

      if np.size(akb1)<=1:
        z1 = np.polyfit(yy, zz, 2)  # 曲线拟合，返回值为多项式的各项系数
        zz1 = np.polyfit(yyy, zzz, 2)  # 曲线拟合，返回值为多项式的各项系数

      elif np.size(akb1)>1:
        z1 = np.polyfit(y_original, z_original, 4)  # 曲线拟合，返回值为多项式的各项系数
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

      y_out = y_adjusted[0:len(y_adjusted) // 3]
      y_out = np.hstack((y_out, y_adjusted[len(y_adjusted) * 2 // 3:]))
      z_out = z_adjusted[0:len(z_adjusted) // 3]
      z_out = np.hstack((z_out, z_adjusted[len(z_adjusted) * 2 // 3:]))
      y_inside = y_adjusted[len(y_adjusted)//6:len(y_adjusted) *5// 6]
      z_inside = z_adjusted[len(z_adjusted)//6:len(z_adjusted) *5// 6]

      z1 = np.polyfit(y_inside, z_inside, 2)  # 曲线拟合，返回值为多项式的各项系数
      zz1 = np.polyfit(y_out, z_out, 2)  # 曲线拟合，返回值为多项式的各项系数
      p1 = np.poly1d(z1)  # 返回值为多项式的表达式，也就是函数式子
      pp1 = np.poly1d(zz1)  # 返回值为多项式的表达式，也就是函数式子
      # y_pred = p1(x)  # 根据函数的多项式表达式，求解 y
      # yy_pred = pp1(x)  # 根据函数的多项式表达式，求解 y
      z2 = np.asarray(z1)
      zz2 = np.asarray(zz1)



      '''
      xxx = x[0:len(x) // 3]
      xxx = np.hstack((xxx, x[len(x) * 2 // 3:]))
      yyy = y[0:len(y) // 3]
      yyy = np.hstack((yyy, y[len(y) * 2 // 3:]))
      xx = x[len(x) // 6:len(x) *5// 6]
      yy = y[len(y) // 6:len(y) *5// 6]
      z1 = np.polyfit(xx, yy, 2)  # 曲线拟合，返回值为多项式的各项系数
      zz1 = np.polyfit(xxx, yyy, 2)  # 曲线拟合，返回值为多项式的各项系数
      p1 = np.poly1d(z1)  # 返回值为多项式的表达式，也就是函数式子
      pp1 = np.poly1d(zz1)  # 返回值为多项式的表达式，也就是函数式子

      y_pred = p1(x)  # 根据函数的多项式表达式，求解 y
      yy_pred = pp1(x)  # 根据函数的多项式表达式，求解 y
      z2 = np.asarray(z1)
      zz2 = np.asarray(zz1)
      '''



      #
      #
      #

      if (tank-xmin-slicing_min-1 <=2 and tank-xmin-slicing_min-1>=-2):#因为拐点的范围比较大，比0.2mm要大得多，所以如果这里用0.2mm的话，那么这边边上一片邻域都是和它接近一模一样的拐点。
        start_guai=1
        # save_plt="test/FinalOutPut/{}/".format(afile)
        # plt.savefig(save_plt+"Filter_One_{}.png".format(tank1))
        # plt.clf()
        # 保存拐点绿色中间的
        #TODO:把数据保存方式进行改进，训练模型的过程
        #middle_fitting 和side_fitting 保存数据最后可以用return方式返回
        data_list = []
        data_list.append(z2[0])
        data_list.append(z2[1])
        data_list.append(z2[2])
        # list1 = json.dumps(data_list)
        # f33.write(list1 + "\n")
        # 保存拐点红色两边的
        data_list = []
        data_list.append(zz2[0])
        data_list.append(zz2[1])
        data_list.append(zz2[2])
        # data_list.append(z2[3])
        # list2 = json.dumps(data_list)
        # f44.write(list2 + "\n")
      else:
        if(start_guai==0):
          #保存正常绿色中间的
          data_list = []
          data_list.append(z2[0])
          data_list.append(z2[1])
          data_list.append(z2[2])
          # list1 = json.dumps(data_list)
          # f11.write(list1 + "\n")
          middle_fitting.append(data_list)
          # 保存正常红色两边的
          data_list = []
          data_list.append(zz2[0])
          data_list.append(zz2[1])
          data_list.append(zz2[2])
          # list2 = json.dumps(data_list)
          # f22.write(list2 + "\n")
          side_fitting.append(data_list)
        elif(start_guai==1):
          # 保存拐点之后绿色中间的
          data_list = []
          data_list.append(z2[0])
          data_list.append(z2[1])
          data_list.append(z2[2])
          # list1 = json.dumps(data_list)
          # f55.write(list1 + "\n")
          # 保存拐点之后红色两边的
          data_list = []
          data_list.append(zz2[0])
          data_list.append(zz2[1])
          data_list.append(zz2[2])
          # list2 = json.dumps(data_list)
          # f66.write(list2 + "\n")


        # plt.savefig(save_path.format(tank1))
        # plt.clf()
        cc = 0

      # poly3.append(z2[0])
      # poly2.append(z2[1])
      # poly1.append(z2[2])
      print(tank1)

  result = np.array(list_all)
  bstart = time.time()
  print(bstart - astart)
  # pcd_vector = o3d.geometry.PointCloud()
  # 加载点坐标
  # pcd_vector.points = o3d.utility.Vector3dVector(result[:, :3])
  #
  # o3d.visualization.draw_geometries([pcd_vector])
  middle_fit = np.array(middle_fitting)
  side_fit = np.array(side_fitting)
  AAA=np.mean(middle_fit[:, 2])
  AAA1 = np.mean(side_fit[:, 2])
  AAAA=np.mean(middle_fit[:,1])
  # AAAAA = np.mean(middle_fit[:,2])
  # print(AAA)
  # print(AAAA)


  return middle_fit,side_fit


if __name__ == "__main__":

  train_coefficient()

