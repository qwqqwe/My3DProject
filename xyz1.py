import time

import cv2
import scipy.linalg as linalg
import math
import scipy
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from mayavi import mlab

def viz_mayavi(points):
    x = points[:, 0]  # x position of point
    y = points[:, 1]  # y position of point
    z = points[:, 2]  # z position of point
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
    mlab.points3d(x, y, z,
                          y,          # Values used for Color
                          mode="point",
                          colormap='spectral', # 'bone', 'copper', 'gnuplot'
                          # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                          figure=fig,
                          )
    mlab.show()

def Point_Show(pca_point_cloud):
    x = []
    y = []
    pca_point_cloud = np.asarray(pca_point_cloud)
    for i in range(len(pca_point_cloud)):
        x.append(pca_point_cloud[i][0])
        y.append(pca_point_cloud[i][1])
    plt.scatter(x,y)
    plt.show()

def plane_param(point_cloud_vector,point):
    """
    不共线的三个点确定一个平面
    :return: 平面方程系数:a,b,c,d
    """
    #n1 = point_cloud_vector / np.linalg.norm(point_cloud_vector)  # 单位法向量
    n1=point_cloud_vector
    #print(n1)
    A = n1[0]
    B = n1[1]
    C = n1[2]
    D = -A * point[0] - B * point[1] - C * point[2]
    return A, B, C, D


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

def PCA(data, correlation=False, sort=True):
    # normalize 归一化
    mean_data = np.mean(data, axis=0)
    normal_data = data - mean_data
    # 计算对称的协方差矩阵
    H = np.dot(normal_data.T, normal_data)
    # SVD奇异值分解，得到H矩阵的特征值和特征向量
    eigenvectors, eigenvalues, _ = np.linalg.svd(H)

    if sort:
        sort = eigenvalues.argsort()[::-1]    #对特征向量进行排序，从大到小，返回索引值
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors

def Point_Cloud_Show(points):
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], cmap='spectral', s=2, linewidths=0, alpha=1, marker=".")
    plt.title('Point Cloud')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def Change_Show(points):
    x = []
    y = []
    pca_point_cloud = np.asarray(points)
    for i in range(len(pca_point_cloud)):
        x.append(pca_point_cloud[i][0])
        y.append(pca_point_cloud[i][1])
    plt.scatter(x,y)
    plt.show()


def downsample(point_cloud_vector,pcd):
    # 将原数据进行降维度处理
    point_cloud_encode = (np.dot(point_cloud_vector.T, pcd.T)).T  # 主成分的转置 dot 原数据
    Point_Show(point_cloud_encode)
    point_cloud_decode = (np.dot(point_cloud_vector,point_cloud_encode.T)).T
    #Point_Cloud_Show(point_cloud_decode)
    point_cloud_raw1=np.asarray(pcd)
    #Point_Cloud_Show(point_cloud_raw1)
    pcd_raw = o3d.geometry.PointCloud()
    pcd_raw.points = o3d.utility.Vector3dVector(point_cloud_raw1)
    pcd_decode = o3d.geometry.PointCloud()
    pcd_decode.points = o3d.utility.Vector3dVector(point_cloud_decode)


# 定义平面方程Ax+By+Cz+D=0
# 以z=0平面为例，即在xy平面上的投影A=0, B=0, C=1(任意值), D=0
# para[0, 0, 1, 0]
def point_project(points, para):
    x = points[:, 0]  # x position of point
    y = points[:, 1]  # y position of point
    z = points[:, 2]  # z position of point
    d = para[0] ** 2 + para[1] ** 2 + para[2] ** 2
    t = -(para[0] * x + para[1] * y + para[2] * z + para[3]) / d
    x = para[0] * t + x
    y = para[1] * t + y
    z = para[2] * t + z
    return np.array([x, y, z]).T


#矩阵写法
#定义平面方程Ax+By+Cz+D=0
#以z=0平面为例，即在xy平面上的投影A=0, B=0, C=1(任意值), D=0
#para[0, 0, 1, 0]
def point_project_array(points, para):
    para  = np.array(para)
    d = para[0]**2 + para[1]**2 + para[2]**2
    t = -(np.matmul(points[:, :3], para[:3].T) + para[3])/d
    points = np.matmul(t[:, np.newaxis], para[np.newaxis, :3]) + points[:, :3]
    return points



def point_project_array(points, para):
    #旋转？
    para  = np.array(para)
    d = para[0]**2 + para[1]**2 + para[2]**2
    t = -(np.matmul(points[:, :3], para[:3].T) + para[3])/d
    points = np.matmul(t[:, np.newaxis], para[np.newaxis, :3]) + points[:, :3]
    return points

def rotate_mat(axis, radian):
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    return rot_matrix

def Router(v):
    #求向量V与标准xyz坐标的角度
    x1=np.array((1,0,0))
    y1=v[:, 0]
    x2=np.array((0,1,0))
    y2=v[:, 1]
    x3 = np.array((0, 0, 1))
    y3 =v[:, 2]
    l_x1 = np.sqrt(x1.dot(x1))
    l_y1 = np.sqrt(y1.dot(y1))
    dian1 = x1.dot(y1)  #x1点积y1
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
    return angle_hu1,angle_hu2,angle_hu3


def NamePoint(points):
    x = []
    y = []
    pca_point_cloud = np.asarray(points)
    for i in range(len(pca_point_cloud)):
        x.append(pca_point_cloud[i][0])
        y.append(pca_point_cloud[i][1])

    n=np.arange(len(x))
    fig, ax = plt.subplots()
    ax.scatter(x, y, c='r')

    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))
    plt.show()

def noft():
    pass



if __name__ == '__main__':
    txt_path = 'txtcouldpoint/Depth_L5000_t220802_104306_01111.txt'
    start_time = time.time()
    # 通过numpy读取txt点云
    pcd_1 = np.genfromtxt(txt_path, delimiter=",")
    pcd = o3d.geometry.PointCloud()
    print(pcd_1.shape)

    # 加载点坐标
    pcd.points = o3d.utility.Vector3dVector(pcd_1)

    # pcd_vector = pcd_vector.select_by_index(np.where(pcd[:, 2] <= y_threshold)[0])

    end_time = time.time()
    print(end_time - start_time)
    #下采样
    pcd=pcd.uniform_down_sample(50)
    #pcd = pcd.random_down_sample(0.01)

    pcd = pcd.translate((0,0,0), relative=False)
    # pcd_new = copy.deepcopy(pcd_tx).translate((0.2, 0.2, 0.2),False)  # relative 可以省略
    pcd.paint_uniform_color([0, 1, 0])
    #pcd = o3d.io.read_point_cloud("drill/Cylinder.pcd")
    #points = np.asarray(pcd.points)
    points=pcd.points


    #导入完成#



    # 用PCA分析点云主方向
    w, v = PCA(points) # PCA方法得到对应的特征值和特征向量
    point_cloud_vector = v[:, 0] #点云主方向对应的向量为最大特征值对应的特征向量
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    # print('v',v)
    # 三个特征向量组成了三个坐标轴
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame().rotate(v, center=(0, 0, 0))
    pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
    # # 可视化
    o3d.visualization.draw_geometries([pc_view, axis], point_show_normal=True)
    point = np.asarray(pcd.points)  #转换为数组
    #downsample(v, point)


    # vector = np.mat(v[:, 0:2])
    # # vector_transpose => (2,3)
    # vector_transpose = vector.transpose()
    # # pca_point_cloud_1 => (10000, 2)
    # pca_point_cloud_1 = np.dot(dat, vector)
    # print(pca_point_cloud_1)
    # # 3、PCA降维之后成分还原显示
    # Point_Show(pca_point_cloud_1)
    # # pca_point_cloud_1 => (10000, 3)
    # pca_point_cloud_2 = np.dot(pca_point_cloud_1, vector_transpose)
    # Point_Cloud_Show(pca_point_cloud_2)




    # 1.获取位于点云上的三个点P1,P2,P3


    #计算臂展
    # ppp=np.dot(point_cloud_vector, point.T)
    # #print(ppp.shape)
    # ppp_max=ppp.max()
    # ppp_min=ppp.min()
    # ppp_value=ppp_max-ppp_min
    #print('aaa',ppp_max*point_cloud_vector,ppp_min*point_cloud_vector)
    #print(ppp_min,ppp_max)
    #print(2.3294797829924336*point_cloud_vector)

    #-63.824439332442125 65.42484338162815
    astart=time.time()
    # 计算要切割的值
    P2 = np.array([0, 0, 0])#zyx
    # a, b, c, d = plane_param(point_cloud_vector,P2)
    a, b, c, d = plane_param(v[:, 1], P2)
    point_size = point.shape[0]
    idx = []
    # 3.设置切片厚度阈值，此值为切片厚度的一半
    Delta = 0.1
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
    if (slicing_min>slicing_max):
        slicing_min=slicing_points[-1][0]
        slicing_max=slicing_points[0][0]
    bstart=time.time()
    print("Start",bstart-astart)
    print(slicing_min,slicing_max)

    # mask =point[:, 0] < slicing_max
    # pcd.points = o3d.utility.Vector3dVector(point[mask])
    # points2=np.asarray(pcd.points)
    # mask2=points2[:, 0] > slicing_min
    # pcd.points = o3d.utility.Vector3dVector(points2[mask2])
    #visualizer_cloud(pcd)

    # 2.计算P1,P2,P3三点确定的平面，以此作为切片
    P2 = np.array([0, 0, 0])#zyx
    # a, b, c, d = plane_param(point_cloud_vector,P2)
    a, b, c, d = plane_param(v[:, 0], P2)
    point_size = point.shape[0]
    idx = []
    # 3.设置切片厚度阈值，此值为切片厚度的一半
    Delta = 0.1
    # 4.循环迭代查找满足切片的点
    for i in range(point_size):
        Wr = a * point[i][0] + b * point[i][1] + c * point[i][2] + d - Delta
        Wl = a * point[i][0] + b * point[i][1] + c * point[i][2] + d + Delta
        if Wr * Wl <= 0:
            idx.append(i)
    # 5.提取切片点云
    slicing_cloud = (pcd.select_by_index(idx))
    slicing_points = np.asarray(slicing_cloud.points)
    #print(slicing_points[0],slicing_points[-1])
    #print(slicing_points)




    ################################
    project_pane = [a, b, c, d]
    points_new = point_project_array(slicing_points, project_pane)
    #print(slicing_points.shape)
    #Point_Show(points_new)
    #Point_Cloud_Show(points_new)
    Point_Show(points_new)
    pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points_new))
    # visualizer_cloud(pc_view)
    #o3d.visualization.draw_geometries([pc_view,slicing_cloud])
    ################################
    #转化xy轴
    h1,h2,h3=Router(v)
    R1 = pcd.get_rotation_matrix_from_xyz((h1,h2,h3))
    R2 = pcd.get_rotation_matrix_from_xyz((0,np.pi/2,0))
    pc_view.rotate(R1)
    #pc_view.rotate(R2)
    visualizer_cloud(pc_view)
    poi = np.asarray(pc_view.points)
    print(poi)
    #Point_Show(poi)
    #NamePoint(poi)
    ################################################################################################
    #转换排序
    poi_x=poi[:, 0]
    pre_sort_x=sorted(enumerate(poi_x), key=lambda poi_x:poi_x[1])
    sorted_poi=np.zeros((poi.shape))
    for i in range(len(poi_x)):
        sorted_poi[i]=poi[pre_sort_x[i][0]]

    X_A=sorted_poi[0]
    X_E=sorted_poi[-1]
    X_C=sorted_poi[0]
    Max_C=0
    Ki_left=[0,0]
    Ki_right=[len(poi_x),0]
    for number_i,i in enumerate(sorted_poi):
        if (i[1]>X_C[1]):
            X_C=i
            Max_C=number_i
    for i in range(0,Max_C):
        kkk=(sorted_poi[Max_C][1] - sorted_poi[i][1]) / (sorted_poi[Max_C][0] - sorted_poi[i][0])
        if(kkk>Ki_left[1]):
            Ki_left[0]=i
            Ki_left[1]=kkk
    X_B=sorted_poi[Ki_left[0]]
    for i in range(Max_C+1,len(poi_x)):
        kkk=(sorted_poi[Max_C][1] - sorted_poi[i][1]) / (sorted_poi[Max_C][0] - sorted_poi[i][0])
        if(kkk<Ki_right[1]):
            Ki_right[0]=i
            Ki_right[1]=kkk
    X_D=sorted_poi[Ki_right[0]-1]
    print(X_A,X_B,X_C,X_D,X_E)
    #NamePoint(sorted_poi)
    Point_Show(sorted_poi)
    # 6.可视化切片
    #visualizer_cloud(slicing_cloud)
# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证







