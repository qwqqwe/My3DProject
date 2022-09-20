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

def visualizer_cloud(filtered):
  # ------------------------显示点云切片---------------------------
  vis = o3d.visualization.Visualizer()
  vis.create_window(window_name='点云切片', width=800, height=600)
  # -----------------------可视化参数设置--------------------------
  opt = vis.get_render_option()
  opt.background_color = np.asarray([0, 0, 0])  # 设置背景色*****
  opt.point_size = 1  # 设置点的大小*************
  vis.add_geometry(filtered)  # 加载点云到可视化窗口
  vis.run()  # 激活显示窗口，这个函数将阻塞当前线程，直到窗口关闭。
  vis.destroy_window()  # 销毁窗口，这个函数必须从主线程调用。

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


def displayu(point_size,tank1,kkk):
  if(tank1<63-kkk):
    return point_size*((tank1-1)+kkk)//63
  else:
    return point_size

def displaynou(point_size, tank1,kkk):
  if(tank1>kkk):
    return point_size*((tank1-1)-kkk)//63
  else:
    return 0
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

# v [[-9.99799735e-01 -2.00116460e-02  1.57309124e-04]
#  [ 2.00120025e-02 -9.99715558e-01  1.29739226e-02]
#  [-1.02365167e-04  1.29744724e-02  9.99915823e-01]]


# v [[-9.99585167e-01 -2.87980184e-02  4.10129614e-04]
#  [ 2.87918022e-02 -9.98808086e-01  3.94136860e-02]
#  [-7.25395281e-04  3.94091443e-02  9.99222895e-01]]

def mean_std(input):
  mm=np.mean(input)
  s = np.std(input)
  return mm,s





# @func_line_time
# 定义一个测试函数
def display():
  a=1
  # txt_path = '../txtcouldpoint/Third_{}.txt'.format(a)
  # fp = open('AllOutPutNom/O{}/1.txt'.format(a), 'w')
  # #save_path="C:/Users/Administrator/PycharmProjects/My3DProject/test/AllOutPutNom/O{}/Filter_{}.png"
  # saveeeee_path="C:/Users/Administrator/PycharmProjects/My3DProject/test/AllOutPutNom/O{}/".format(a)
  # save_path=saveeeee_path+"Filter_{}.png"
  txt_path='../txtcouldpoint/Third_6.txt'
  # fp=open('AllOutPutNom/O777/1.txt', 'w')
  # fp = open('AllOutPutNom/O777/2.txt', 'w')
  # save_path="../test/AllOutPutNom/O777/Filter_{}.png"
  fp = open('AllOutPutNom/O10/2.txt', 'w')
  f3 = open('AllOutPutNom/O10/3.txt', 'a+')
  f4 = open('AllOutPutNom/O10/4.txt', 'a+')

  save_path = "../test/AllOutPutNom/O10/Filter_{}.png"

  # np.set_printoptions(precision=5)
  np.set_printoptions(formatter={'float': '{: 0.5f}'.format})


  start_time = time.time()
  # 通过numpy读取txt点云
  pcd_1 = np.genfromtxt(txt_path, delimiter=",")
  pcd = o3d.geometry.PointCloud()
  print(pcd_1.shape)

  # 加载点坐标
  pcd.points = o3d.utility.Vector3dVector(pcd_1)

  end_time = time.time()
  print(end_time - start_time)

  pcd = pcd.translate((0, 0, 0), relative=False)  #平移


  # 用PCA分析点云主方向
  w, v = xyz1.PCA(pcd.points)  # PCA方法得到对应的特征值和特征向量
  point_cloud_vector = v[:, 0]  # 点云主方向对应的向量为最大特征值对应的特征向量
  print('the main orientation of this pointcloud is: ', point_cloud_vector)
  print('v',v)
  if(v[0][0]<0):
    v[:,0]=-v[:,0]

  # 三个特征向量组成了三个坐标轴
  # axis = o3d.geometry.TriangleMesh.create_coordinate_frame().rotate(v, center=(0, 0, 0))
  # pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pcd.points))
  # # 可视化
  # o3d.visualization.draw_geometries([pc_view, axis], point_show_normal=True)

  # 下采样
  pcd = pcd.uniform_down_sample(50) #均匀下采样，50个点取一个点
  # pcd = pcd.random_down_sample(0.02)
  # point = np.asarray(pcd.points)
  # xyz1.visualizer_cloud(pcd)

  # pcd.paint_uniform_color([0, 1, 0])
  # points = pcd.points
  #
  # #加载完成

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
  # akkk=[]
  # for i in ind:
  #   print(pcd.points[i][2])
  #   akkk.append(pcd.points[i][2])
  #
  # akkk=np.asarray(akkk)
  # print(akkk.min(),akkk.max(),akkk.mean())
  # 可视化统计滤波后的点云和噪声点云
  # display_inlier_outlier(pcd, ind)

  pcd = sor_pcd
  points = pcd.points
  point = np.asarray(points)

  # -62.235174832472566 63.63087516752742

  # 计算要切割的值
  P2 = np.array([0, 0, 0])  # zyx
  # a, b, c, d = plane_param(point_cloud_vector,P2)
  a, b, c, d = plane_param(v[:, 1], P2)
  point_size = point.shape[0]
  idx = []
  # 3.设置切片厚度阈值，此值为切片厚度的一半
  Delta = 0.05


  # 4.循环迭代查找满足切片的点
  for i in range(point_size):
    Wr = a * point[i][0] + b * point[i][1] + c * point[i][2] + d - Delta
    Wl = a * point[i][0] + b * point[i][1] + c * point[i][2] + d + Delta
    if Wr * Wl <= 0:
      idx.append(i)
  # 5.提取切片点云
  slicing_cloud = (pcd.select_by_index(idx))
  slicing_points = np.asarray(slicing_cloud.points)
  slicing_min = slicing_points[0][0]
  slicing_max = slicing_points[-1][0]
  if (slicing_min > slicing_max):
    slicing_min = slicing_points[-1][0]
    slicing_max = slicing_points[0][0]

  # 转化xy轴
  h1, h2, h3 = xyz1.Router(v)
  R1 = pcd.get_rotation_matrix_from_xyz((h1, h2, h3))
  R2 = pcd.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))
  slicing_cloud.rotate(R1)        # 旋转
  slicing_cloud.rotate(R2)
  poi = np.asarray(slicing_cloud.points)  #转换数组

  # axis = o3d.geometry.TriangleMesh.create_coordinate_frame().rotate(v, center=(0, 0, 0))
  # pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(slicing_cloud.points))
  # # 可视化
  # o3d.visualization.draw_geometries([pc_view, axis], point_show_normal=True)



  # xyz1.Point_Show(poi)
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
    if now_length > sorted_poi[-1][0]-sorted_poi[0][0]-4:  #去掉最后4mm，减少误差
      sorrrayiey = i    #标志位
      break



  #print(sorrrayiex,len(sorted_poi))

  x = sorted_poi[:, 0]
  #print(x)
  # print(sorted_poi,sorted_poi)
  y = sorted_poi[:, 1]
  x = x - x[0]
  y = y - y[0]


  a=time.time()

  # z1 = np.polyfit(x[sorrrayiex:sorrrayiey], y[sorrrayiex:sorrrayiey], 15)  # 曲线拟合，返回值为多项式的各项系数

  # p1 = np.poly1d(z1)  # 返回值为多项式的表达式，也就是函数式子
  # print(p1)
  # y_pred = p1(x[sorrrayiex:sorrrayiey])  # 根据函数的多项式表达式，求解 y
  # print(np.polyval(p1, 29))             #根据多项式求解特定 x 对应的 y 值
  # print(np.polyval(z1, 29))             #根据多项式求解特定 x 对应的 y 值
  #kneedle = KneeLocator(x[sorrrayiex:], y[sorrrayiex:], S=4, curve='concave', direction='decreasing',online=True)


  plt.plot(x[sorrrayiex:sorrrayiey], y[sorrrayiex:sorrrayiey], '*', label='original values')
  # print(np.polyval(p1, kneedle.elbow))
  # kkk=time.time()
  akb=signal.argrelmin(y[sorrrayiex:sorrrayiey], order=15)    #局部相对最小
  print("akb", akb[0][0])
  xmin=x[akb[0][0]+sorrrayiex]
  ymin=y[akb[0][0]+sorrrayiex]
  for i in akb[0]:
    # print(i)
    if (y[i+sorrrayiex]<ymin):
      ymin=y[i+sorrrayiex]
      xmin=x[i+sorrrayiex]

  # kkb=time.time()
  # print('Knee-Locator time: ', kkb-kkk)
  #plt.scatter(kneedle.elbow,kneedle.elbow_y, s=100, c='r', marker='*', alpha=0.65)
  # plt.plot(x[akb[0]+sorrrayiex], y[akb[0]+sorrrayiex], '+',markersize=20)  # 极小值点
  plt.plot(xmin,ymin, '+',markersize=20)  # 极小值点
  # plt.plot(x[signal.argrelextrema(-y_pred, np.greater)[0]+sorrrayiex], y_pred[signal.argrelextrema(-y_pred, np.greater)], '+',markersize=10)

  # plt.scatter(kneedle.elbow,np.polyval(kneedle.elbow),c='r')
  # plt.plot(x[sorrrayiex:sorrrayiey], y_pred, 'r', label='fit values')
  plt.title('')
  plt.xlabel('')
  plt.ylabel('')
  plt.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))
  b=time.time()
  print('time',b-a)
  plt.show()





  # x = sorted_poi[:, 0]
  # y = sorted_poi[:, 1]
  # x = x - x[0]
  # y = y - y[0]
  #
  # plt.plot(x, y, '*', label='original values')
  # plt.title('')
  # plt.xlabel('')
  # plt.ylabel('')
  # plt.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))
  # plt.show()
  # plt.savefig(save_path.format("0"))
  # plt.clf()


  # return 0



  # Point_Show(slicing_points)
  # print(slicing_min, slicing_max)
  mask =point[:, 0] < slicing_max     #比较point第一列与slicing_max，结果保存在mask
  pcd.points = o3d.utility.Vector3dVector(point[mask])
  points2=np.asarray(pcd.points)
  mask2=points2[:, 0] > slicing_min
  pcd.points = o3d.utility.Vector3dVector(points2[mask2])
  point= np.asarray(pcd.points)

  # axis = o3d.geometry.TriangleMesh.create_coordinate_frame().rotate(v, center=(0, 0, 0))
  # pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pcd.points))
  # # 可视化
  # o3d.visualization.draw_geometries([pc_view, axis], point_show_normal=True)
  #visualizer_cloud(pcd)

  # return 0

  # 2.计算P1,P2,P3三点确定的平面，以此作为切片
  # tank1 = -62.23 + 1

  poly3 = list()
  poly2 = list()
  poly1 = list()
  poly0 = list()

  tank1=1   #每次切间隔距离（除以10为真实距离单位：mm）
  astart = time.time()
  # print(point)
  while (slicing_min + 1+tank1*2/10 < slicing_max - 0.1):
    tank=slicing_min + 1+tank1*2/10      #tank切的位置
    P2 = np.array([tank, 0, 0])  # xyz
    # a, b, c, d = plane_param(point_cloud_vector,P2)
    a, b, c, d = xyz1.plane_param(v[:, 0], P2)
    point_size = point.shape[0]
    idx = []
    # 3.设置切片厚度阈值，此值为切片厚度的一半
    Delta = 0.1
    # 4.循环迭代查找满足切片的点
    for i in range(point_size):
    # for i in range(displaynou(point_size,tank1,6),displayu(point_size,tank1,3)):
      Wr = a * point[i][0] + b * point[i][1] + c * point[i][2] + d - Delta
      Wl = a * point[i][0] + b * point[i][1] + c * point[i][2] + d + Delta
      if Wr * Wl <= 0:
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
    # xyz1.visualizer_cloud(pc_view)
    # 转化xy轴
    h1, h2, h3 = xyz1.Router(v)
    R1 = pcd.get_rotation_matrix_from_xyz((h1, h2, h3))
    R2 = pcd.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))
    # pc_view.rotate(R1)

    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame().rotate(v, center=(0, 0, 0))
    # pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc_view.points))
    # # 可视化
    # o3d.visualization.draw_geometries([pc_view, axis], point_show_normal=True)

    # pc_view.rotate(R2)
    # xyz1.visualizer_cloud(pc_view)
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
    # print(z1)
    y_pred = p1(x)  # 根据函数的多项式表达式，求解 y
    # print(np.polyval(p1, 29))             #根据多项式求解特定 x 对应的 y 值
    # print(np.polyval(z1, 29))             #根据多项式求解特定 x 对应的 y 值
    z2 = np.asarray(z1)

    # print("akb", akb1[0][0])
    # xmax=x[akb1]
    #
    # plt.plot(xmax, ymax, '+', markersize=20)

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
      # plt.savefig("../test/AllOutPutNom/O8/Filter_One_{}.png".format(tank1))
      plt.savefig("../test/AllOutPutNom/O10/Filter_One_{}.png".format(tank1))
      plt.clf()
    else:
      if np.size(akb1)<=1:
        #保存3次拟合的各项系数
        data_list = []
        data_list.append(z2[0])
        data_list.append(z2[1])
        data_list.append(z2[2])
        data_list.append(z2[3])
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
    poly0.append(z2[3])

    print(z2, file=fp)
    tank1 += 1
    print(tank1)


  bstart = time.time()
  print(bstart - astart)
  fp.close()
  f3.close()
  f4.close()

  # polynomial_coefficient3 = np.asarray(poly3)
  # polynomial_coefficient2 = np.asarray(poly2)
  # polynomial_coefficient1 = np.asarray(poly1)
  # polynomial_coefficient0 = np.asarray(poly0)
  #
  # mean3,std3 =mean_std(polynomial_coefficient3)
  # mean2, std2 = mean_std(polynomial_coefficient2)
  # mean1, std1 = mean_std(polynomial_coefficient1)
  # mean0, std0 = mean_std(polynomial_coefficient0)
  #
  # print("mean3 =",mean3,"std3 =",std3)
  # print("mean2 =",mean2,"std2 =",std2)
  # print("mean1 =",mean1,"std1 =",std1)
  # print("mean0 =",mean0,"std0 =",std0)

  # print(mean3, std3)
  # print(mean3, std3)
  # print(mean3, std3)





if __name__ == "__main__":

  display()

