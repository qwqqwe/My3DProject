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

# heartrate.trace(browser=True)
if __name__ == "__main__":

  txt_path = '../txtcouldpoint/Depth_L5000_t220802_104306_01111.txt'
  start_time = time.time()
  # 通过numpy读取txt点云
  pcd_1 = np.genfromtxt(txt_path, delimiter=",")
  pcd = o3d.geometry.PointCloud()
  print(pcd_1.shape)

  # 加载点坐标
  pcd.points = o3d.utility.Vector3dVector(pcd_1)

  end_time = time.time()
  print(end_time - start_time)


  pcd = pcd.translate((0, 0, 0), relative=False)



  # 用PCA分析点云主方向
  w, v = xyz1.PCA(pcd.points)  # PCA方法得到对应的特征值和特征向量
  point_cloud_vector = v[:, 0]  # 点云主方向对应的向量为最大特征值对应的特征向量
  print('the main orientation of this pointcloud is: ', point_cloud_vector)
  # print('v',v)
  # 三个特征向量组成了三个坐标轴
  # axis = o3d.geometry.TriangleMesh.create_coordinate_frame().rotate(v, center=(0, 0, 0))
  # pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pcd.points))
  # # 可视化
  #o3d.visualization.draw_geometries([pc_view, axis], point_show_normal=True)



  # 下采样
  pcd = pcd.uniform_down_sample(50)
  #pcd = pcd.random_down_sample(0.02)
  #point = np.asarray(pcd.points)
  #xyz1.visualizer_cloud(pcd)
  visualizer_cloud(pcd)

  #pcd.paint_uniform_color([0, 1, 0])
  #points = pcd.points
  #
  # #加载完成


  # ------------------------- 统计滤波 --------------------------
  print("->正在进行统计滤波...")
  astart=time.time()
  num_neighbors = 20  # K邻域点的个数
  std_ratio = 2.0  # 标准差乘数
  # 执行统计滤波，返回滤波后的点云sor_pcd和对应的索引ind
  sor_pcd, ind = pcd.remove_statistical_outlier(num_neighbors, std_ratio)
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
  #display_inlier_outlier(pcd, ind)

  pcd=sor_pcd
  points=pcd.points
  point=np.asarray(points)



  #-62.235174832472566 63.63087516752742

  # 2.计算P1,P2,P3三点确定的平面，以此作为切片
  tank1=-62.23+1
  astart = time.time()
  print(point)
  while (tank1<63.63-0.1):
    P2 = np.array([tank1, 0, 0])  # xyz
    # a, b, c, d = plane_param(point_cloud_vector,P2)
    a, b, c, d = xyz1.plane_param(v[:, 0], P2)
    point_size = point.shape[0]
    idx = []
    # 3.设置切片厚度阈值，此值为切片厚度的一半
    Delta = 0.5




    # 4.循环迭代查找满足切片的点
    for i in range(point_size):
      Wr = a * point[i][0] + b * point[i][1] + c * point[i][2] + d - Delta
      Wl = a * point[i][0] + b * point[i][1] + c * point[i][2] + d + Delta
      if Wr * Wl <= 0:
        idx.append(i)




    # 5.提取切片点云
    slicing_cloud = (pcd.select_by_index(idx))
    slicing_points = np.asarray(slicing_cloud.points)

    project_pane = [a, b, c, d]
    points_new = xyz1.point_project_array(slicing_points, project_pane)
    pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points_new))
    #xyz1.Point_Show(points_new)

    # 转化xy轴
    h1, h2, h3 = xyz1.Router(v)
    R1 = pcd.get_rotation_matrix_from_xyz((h1, h2, h3))
    R2 = pcd.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))
    pc_view.rotate(R1)
    # pc_view.rotate(R2)
    poi = np.asarray(pc_view.points)
    #xyz1.Point_Show(poi)
    poi_x = poi[:, 0]
    pre_sort_x = sorted(enumerate(poi_x), key=lambda poi_x: poi_x[1])
    sorted_poi = np.zeros((poi.shape))
    for i in range(len(poi_x)):
      sorted_poi[i] = poi[pre_sort_x[i][0]]


    x = sorted_poi[:,0]
    #print(sorted_poi,sorted_poi)
    y=sorted_poi[:,1]
    x=x-x[0]
    y=y-y[0]


    z1 = np.polyfit(x, y, 5)              # 曲线拟合，返回值为多项式的各项系数
    p1 = np.poly1d(z1)                    # 返回值为多项式的表达式，也就是函数式子
    #print(p1)
    y_pred = p1(x)                        # 根据函数的多项式表达式，求解 y
    #print(np.polyval(p1, 29))             #根据多项式求解特定 x 对应的 y 值
    #print(np.polyval(z1, 29))             #根据多项式求解特定 x 对应的 y 值

  #   plt.plot(x, y, '*', label='original values')
  #   plt.plot(x, y_pred, 'r', label='fit values')
  #   plt.title('')
  #   plt.xlabel('')
  #   plt.ylabel('')
  #   plt.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))
  #   # plt.show()
  #   # saveName=tank1
  #   plt.savefig("C:/Users/Administrator/PycharmProjects/My3DProject/OutPut/Filter_{}.png".format(tank1))
  #   plt.clf()
  #   fp=open('../OutPut/test.txt','a')
  #   print('------------------------------',tank1,'---------------------------------',file=fp)
  #   print(p1,file=fp)
    tank1+=2
  # bstart = time.time()
  # print(bstart - astart)
  # fp.close()
