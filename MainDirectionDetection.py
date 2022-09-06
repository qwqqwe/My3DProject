import time
import cv2
import scipy.linalg as linalg
import math
import scipy
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from mayavi import mlab
if __name__ == '__main__':
    #加载点云
    txt_path = 'txtcouldpoint/Depth_L5000_t220802_104306_01111.txt'
    start_time = time.time()
    pcd_1 = np.genfromtxt(txt_path, delimiter=",")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_1[:, :3])
    #pcd=pcd.uniform_down_sample(2)
    end_time = time.time()
    print(end_time - start_time)
    pcd = pcd.translate((0,0,0), relative=False)
    pcd_1=np.asarray(pcd.points)
    #点云排序
    poi_x=pcd_1[:, 0]
    sorted_poi=pcd_1
    # pre_sort_x=sorted(enumerate(poi_x), key=lambda poi_x:poi_x[1])
    # sorted_poi=np.zeros((pcd_1.shape))
    # for i in range(len(poi_x)):
    #     sorted_poi[i]=pcd_1[pre_sort_x[i][0]]
    oneoften=sorted_poi.shape[0]//10
    nineoften=sorted_poi.shape[0]-oneoften
    ones=[]
    nines=[]
    for i in sorted_poi:
        if i[0]==sorted_poi[oneoften][0]:
            ones.append(i[1])
        if i[0]==sorted_poi[nineoften][0]:
            nines.append(i[1])
    ones=np.array(ones)
    nines=np.array(nines)
    ivalue=(ones.max()+ones.min())/2
    minpoint=[sorted_poi[oneoften][0],ivalue,0]
    jvalue=(nines.max()+nines.min())/2
    maxpoint=[sorted_poi[nineoften][0],jvalue,0]
    valueij=np.array([minpoint,maxpoint])
    print(ones.max()-ones.min(),nines.max()-nines.min())
    v_x=valueij[1]-valueij[0]
    v_x = v_x / linalg.norm(v_x)
    v_z=np.array([0,0,1])
    v_y=np.array([v_x[1],-v_x[0],0])
    v=np.array([v_x,v_y,v_z])
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame().rotate(v, center=(0,0,0))
    pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pcd_1))
    # # 可视化
    o3d.visualization.draw_geometries([pc_view, axis], point_show_normal=True)

    polygon_points = valueij
    lines=[[0,1]]
    color = [[1, 0, 0] for i in range(len(lines))]
    # 添加顶点，点云
    points_pcd = o3d.geometry.PointCloud()
    points_pcd.points = o3d.utility.Vector3dVector(polygon_points)
    points_pcd.paint_uniform_color([0, 0.3, 0])  # 点云颜色

    # 绘制线条
    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.lines = o3d.utility.Vector2iVector(lines)
    lines_pcd.colors = o3d.utility.Vector3dVector(color)  # 线条颜色
    lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='绘制多边形')
    # vis.toggle_full_screen() #全屏

    # 设置
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])  # 背景
    opt.point_size = 1  # 点云大小

    # vis.add_geometry(axis_pcd)
    vis.add_geometry(lines_pcd)
    vis.add_geometry(points_pcd)
    vis.add_geometry(pcd)
    # vis.update_geometry(points)
    vis.run()
    vis.destroy_window()



