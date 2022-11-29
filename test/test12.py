# -*-coding:utf-8 -*-
import numpy as np
import open3d as o3d
import copy

# 在原点创建坐标框架网格
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
# 平移网格
mesh_r = copy.deepcopy(mesh).translate((2, 0, 0))
# 使用欧拉角创建旋转矩阵
mesh_r.rotate(mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, np.pi / 4)), center=(0, 0, 0))
# 可视化
o3d.visualization.draw_geometries([mesh, mesh_r])