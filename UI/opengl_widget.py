from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
from ctypes import *
import open3d as o3d
import time
import xyz1
from matplotlib import pyplot as plt
from scipy import signal
from line_profiler import LineProfiler
from functools import wraps

from PyQt5 import QtCore, QtGui, QtWidgets, QtOpenGL

# txt_path = '..//txtcouldpoint//Finalzhengzheng5.txt'
# pcd = np.loadtxt(txt_path, delimiter=",")
remark =2
# global result11
windll.LoadLibrary("C:/Users/Administrator/Documents/WeChat Files/wxid_bj5u8pz1th8e12/FileStorage/File/2022-08/v2.1.15.138(1)/G56N_SDK_DEMO_2.1.15.138_20220121_1748/CamWrapper/bins/X64/Debug/SgCamWrapper.dll")
# targe1t=windll.LoadLibrary(r"C:\Users\Administrator\Documents\WeChat Files\wxid_bj5u8pz1th8e12\FileStorage\File\2022-08\v2.1.15.138(1)\G56N_SDK_DEMO_2.1.15.138_20220121_1748\CamWrapper\bins\X64\Debug\Dll6.dll")
targe1t=windll.LoadLibrary(r"C:\Users\Administrator\source\repos\Dll6\x64\Debug\Dll6.dll")

def Py_Catch(targe1t):
    targe1t.Catch()
    targe1t.Reportx.restype = POINTER(c_float)
    targe1t.Reporty.restype = POINTER(c_float)
    targe1t.Reportz.restype = POINTER(c_float)
    targe1t.ReportSizeoflen.restype = c_int
    x = targe1t.Reportx()
    y = targe1t.Reporty()
    z = targe1t.Reportz()
    sizeoflen = targe1t.ReportSizeoflen()
    xxx = np.empty([sizeoflen,3], dtype = float)
    xxx[:, 0]=x[:sizeoflen]
    xxx[:, 1]=y[:sizeoflen]
    xxx[:, 2]=z[:sizeoflen]
    targe1t.FreeMemory()
    return xxx
def Py_PrepareToCatch(targe1t):
    a=targe1t.PrepareToCatch()
    return a
def Py_Stop(targe1t):
    a=targe1t.Stop()
    return a

def display_inlier_outlier(cloud, ind):
  inlier_cloud = cloud.select_by_index(ind)
  outlier_cloud = cloud.select_by_index(ind, invert=True)

  print("Showing outliers (red) and inliers (gray): ")
  outlier_cloud.paint_uniform_color([1, 0, 0])
  inlier_cloud.paint_uniform_color([0, 1, 0])
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

def display1(pcd_1):

  # afile='fanzheng5'
  #设fan为我们的正确的方向
  # txt_path= '../txtcouldpoint/Final{}.txt'.format(afile)

  # save_path = "../test/FinalOutPut/{}/".format(afile)+"Filter_{}.png"

  # np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

  start_time = time.time()
  # 通过numpy读取txt点云
  # pcd_1 = np.genfromtxt(txt_path, delimiter=",")
  # pcd_1=np.loadtxt(txt_path, delimiter=",")
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
  temp_point=point
  # np_temp_array=np.array([point.shape[0],4])
  # np_temp_array[:,0]=point[:,0]
  # np_temp_array[:,1]=point[:,1]
  # np_temp_array[:,2]=point[:,2]
  # np_temp_array[:,3]=1



  mesh_1 = o3d.geometry.TriangleMesh.create_coordinate_frame()
  axis = o3d.geometry.TriangleMesh.create_coordinate_frame().rotate(v, center=(0, 0, 0))
  mesh_1.scale(20, center=(0, 0, 0))
  axis.scale(20, center=(0, 0, 0))
  pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pcd.points))
  # 可视化
  # o3d.visualization.draw_geometries([pc_view, axis, mesh_1], point_show_normal=True)

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

  start_guai=0#拐点的判断
  tank1=1   #每次切间隔距离（除以10为真实距离单位：mm）
  astart = time.time()
  step = 0
  indx = 0
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


      for num in range(0,len(y_adjusted)):
        z_pred.append(function(y_adjusted[num]))

      z_adjusted=z_adjusted-np.mean(z_adjusted-z_pred)
      score=Rationality(z_adjusted,z_pred,0.05)

      if(score<=0.9):#得分低的，将重新进行2次拟合进行第二次判断
        z1 = np.polyfit(y_adjusted,z_adjusted,2)
        p1 = np.poly1d(z1)
        z_pred=p1(y_adjusted)
        score=Rationality(z_adjusted,z_pred,0.05)
        if(score<=0.9):
          # print("输出图像")
          step += 1
        else:
          if (step != 0):#输出缺陷位置
            print("缺陷起始位置", tank1 - step)
            print("缺陷结束位置", tank1)
          step = 0

        for i in range(len(z_adjusted)):
          list_1=[]
          if(abs(z_adjusted[i]-z_pred[i])<=0.05):
            list_1.append(x_original[i])
            list_1.append(y_original[i])
            list_1.append(z_original[i])
            list_1.append(1)

            indx += 1
            list_all.append(list_1)

          else:
            list_1.append(x_original[i])
            list_1.append(y_original[i])
            list_1.append(z_original[i])
            list_1.append(0.5)

            indx += 1
            list_all.append(list_1)


      else:
        if (step!=0):#输出缺陷位置
          print("缺陷起始位置",tank1-step)
          print("缺陷结束位置", tank1)
        step = 0

        for i in range(len(z_adjusted)):
          list_1=[]
          list_1.append(x_original[i])
          list_1.append(y_original[i])
          list_1.append(z_original[i])
          list_1.append(1)
          indx += 1
          list_all.append(list_1)


      if(tank+0.2>=slicing_max - 0.1):#焊缝末尾判断输出
        if (step!=0):
          print("缺陷起始位置",tank1-step)
          print("缺陷结束位置", tank1)


      if (tank-xmin-slicing_min-1 <=2 and tank-xmin-slicing_min-1>=-2):#因为拐点的范围比较大，比0.2mm要大得多，所以如果这里用0.2mm的话，那么这边边上一片邻域都是和它接近一模一样的拐点。
        start_guai=1
        None
      else:
        None


      print(tank1)

  result11 = np.array(list_all)
  bstart = time.time()
  print(bstart - astart)

  # return result11
  return temp_point

def display2(pcd_1):

  # afile='fanzheng5'
  #设fan为我们的正确的方向
  # txt_path= '../txtcouldpoint/Final{}.txt'.format(afile)

  # save_path = "../test/FinalOutPut/{}/".format(afile)+"Filter_{}.png"

  # np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
  defect_meassage=[]
  start_time = time.time()
  # 通过numpy读取txt点云
  # pcd_1 = np.genfromtxt(txt_path, delimiter=",")
  # pcd_1=np.loadtxt(txt_path, delimiter=",")
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
  temp_point=point
  # np_temp_array=np.array([point.shape[0],4])
  # np_temp_array[:,0]=point[:,0]
  # np_temp_array[:,1]=point[:,1]
  # np_temp_array[:,2]=point[:,2]
  # np_temp_array[:,3]=1



  mesh_1 = o3d.geometry.TriangleMesh.create_coordinate_frame()
  axis = o3d.geometry.TriangleMesh.create_coordinate_frame().rotate(v, center=(0, 0, 0))
  mesh_1.scale(20, center=(0, 0, 0))
  axis.scale(20, center=(0, 0, 0))
  pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pcd.points))
  # 可视化
  # o3d.visualization.draw_geometries([pc_view, axis, mesh_1], point_show_normal=True)

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

  start_guai=0#拐点的判断
  tank1=1   #每次切间隔距离（除以10为真实距离单位：mm）
  astart = time.time()
  step = 0
  indx = 0
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
      defect_meassage.append("no_data:"+str(tank1))
      # print(tank1, file=f5)
    elif(-1.75<(sorted_poi[-1][1]- sorted_poi[0][1])<1.75):
      no_data = 1
      print("no_data")
      print(tank1)
      # print(tank1, file=f5)
      defect_meassage.append("no_data:" + str(tank1))
    else:
      for i in range(len(poi_x)-1):
        #如果两个点之间差距大于0.5的话，判断点有遗失，数据量不够
        if((sorted_poi[i+1][1]- sorted_poi[i][1])>0.5):
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


      for num in range(0,len(y_adjusted)):
        z_pred.append(function(y_adjusted[num]))

      z_adjusted=z_adjusted-np.mean(z_adjusted-z_pred)
      score=Rationality(z_adjusted,z_pred,0.05)

      if(score<=0.9):#得分低的，将重新进行2次拟合进行第二次判断
        z1 = np.polyfit(y_adjusted,z_adjusted,2)
        p1 = np.poly1d(z1)
        z_pred=p1(y_adjusted)
        score=Rationality(z_adjusted,z_pred,0.05)
        if(score<=0.9):
          # print("输出图像")
          step += 1
        else:
          if (step != 0):#输出缺陷位置
            print("缺陷起始位置", tank1 - step)
            print("缺陷结束位置", tank1)
            defect_meassage.append("缺陷起始位置:" + str(tank1 - step))
            defect_meassage.append("缺陷结束位置:" + str(tank1))
          step = 0

        for i in range(len(z_adjusted)):
          list_1=[]
          if(abs(z_adjusted[i]-z_pred[i])<=0.05):
            list_1.append(x_original[i])
            list_1.append(y_original[i])
            list_1.append(z_original[i])
            list_1.append(1)

            indx += 1
            list_all.append(list_1)

          else:
            list_1.append(x_original[i])
            list_1.append(y_original[i])
            list_1.append(z_original[i])
            list_1.append(0.5)

            indx += 1
            list_all.append(list_1)


      else:
        if (step!=0):#输出缺陷位置
          print("缺陷起始位置",tank1-step)
          print("缺陷结束位置", tank1)
          defect_meassage.append("缺陷起始位置:" + str(tank1 - step))
          defect_meassage.append("缺陷结束位置:" + str(tank1))
        step = 0

        for i in range(len(z_adjusted)):
          list_1=[]
          list_1.append(x_original[i])
          list_1.append(y_original[i])
          list_1.append(z_original[i])
          list_1.append(1)
          indx += 1
          list_all.append(list_1)


      if(tank+0.2>=slicing_max - 0.1):#焊缝末尾判断输出
        if (step!=0):
          print("缺陷起始位置",tank1-step)
          print("缺陷结束位置", tank1)
          defect_meassage.append("缺陷起始位置:" + str(tank1 - step))
          defect_meassage.append("缺陷结束位置:" + str(tank1))


      if (tank-xmin-slicing_min-1 <=2 and tank-xmin-slicing_min-1>=-2):#因为拐点的范围比较大，比0.2mm要大得多，所以如果这里用0.2mm的话，那么这边边上一片邻域都是和它接近一模一样的拐点。
        start_guai=1
        None
      else:
        None


      print(tank1)

  result11 = np.array(list_all)
  bstart = time.time()
  print(bstart - astart)

  # return result11
  return temp_point,defect_meassage

class openGl_widget(QtWidgets.QOpenGLWidget):
    remark = 0

    pcd =None
    # txt_path = '..//txtcouldpoint//Finalzhengzheng5.txt'
    # txt_path = 'txtcouldpoint/Original/Third_146.txt'
    # txt_path = 'heidian.txt'
    # remark = 2
    # start_time = time.time()
    # 通过numpy读取txt点云
    # pcd = np.loadtxt(txt_path, delimiter=",")

    def __init__(self, parent=None):
        super().__init__(parent)
        # remark = 2
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
        print("1")
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # print("2")
        # self.remark += 1
        glBegin(GL_POINTS)
        if self.remark==1:
            print("3")
            c = self.pcd.shape[0]
            print(c)
            # pcd1 = display1()
            # self.pcd=result11
            for i in range(0, c):
                x = (self.pcd[i][0]) / 75
                y = (self.pcd[i][1]) / 75
                z = (self.pcd[i][2]) / 75
                # ccc = self.pcd[i][3]
                ccc = 1
                if ccc == 1:
                    glColor3f(0, 1, 0.0)
                else:
                    glColor3f(1, 0.0, 0.0)
                glVertex3f(x, y, 0)
            # c = self.pcd.shape[0]
            # c = result11.shape[0]
            # print(c)
            # self.remark -= 1
            # glColor3f(1.0, 0.0, 0.0)
            # print("4")
            # for i in range(0, c):
            #     x = (self.pcd[i][0] - 70) / 70
            #     y = self.pcd[i][1] / 70
            #     z = self.pcd[i][2] / 3
            #     # x = (result11[i][0] - 70) / 70
            #     # y = result11[i][1] / 70
            #     # z = result11[i][2] / 3
            #     glColor3f(z, 0.0, 0.0)
            #     glVertex3f(x, y, z)
        glEnd()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(20, w / h, 0.01, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)

    def Pre_ToCatch(self):
        return_prepare = Py_PrepareToCatch(targe1t)
        print('return_prepare', return_prepare)
        # print('return_prepare')

    def ToCatch(self):
        bbb1 = Py_Catch(targe1t)
        print("tocatch")
        # self.pcd = bbb1
        return bbb1


    def ToStop(self):
      Py_Stop(targe1t)
    def change(self):
        self.remark=1

    def cccc(self,bbb1):
        self.pcd=display1(bbb1)

    def dddd(self,bbb1):
        self.pcd,message=display2(bbb1)
        return message