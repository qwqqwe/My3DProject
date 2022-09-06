import open3d as o3d
import numpy as np
import time

print("->正在加载点云... ")
txt_path = '../txtcouldpoint/Depth_L5000_t220802_104306_01111.txt'
pcd_1 = np.genfromtxt(txt_path, delimiter=",")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcd_1[:, :3])
print("原始点云：", pcd)

# ------------------------- 半径滤波 --------------------------
print("->正在进行半径滤波...")
num_points = 20  # 邻域球内的最少点数，低于该值的点为噪声点
radius = 0.05    # 邻域半径大小
# 执行半径滤波，返回滤波后的点云sor_pcd和对应的索引ind
a=time.time()
sor_pcd, ind = pcd.remove_radius_outlier(num_points, radius)
b=time.time()
c=b-a
sor_pcd.paint_uniform_color([0, 0, 1])
print("半径滤波后的点云：", sor_pcd)
sor_pcd.paint_uniform_color([0, 0, 1])
# 提取噪声点云
sor_noise_pcd = pcd.select_by_index(ind,invert = True)
print("噪声点云：", sor_noise_pcd)
sor_noise_pcd.paint_uniform_color([1, 0, 0])
# ===========================================================

# 可视化半径滤波后的点云和噪声点云
#o3d.visualization.draw_geometries([sor_pcd, sor_noise_pcd])



print("->正在进行统计滤波...")
num_neighbors = 10 # K邻域点的个数
std_ratio = 3 # 标准差乘数
# 执行统计滤波，返回滤波后的点云sor_pcd和对应的索引ind
aa=time.time()
sor_pcd, ind = pcd.remove_statistical_outlier(num_neighbors, std_ratio)
bb=time.time()
cc=bb-aa
print(cc-c)
sor_pcd.paint_uniform_color([0, 0, 1])
print("统计滤波后的点云：", sor_pcd)
sor_pcd.paint_uniform_color([0, 0, 1])
# 提取噪声点云
sor_noise_pcd = pcd.select_by_index(ind,invert = True)
print("噪声点云：", sor_noise_pcd)
sor_noise_pcd.paint_uniform_color([1, 0, 0])
# ===========================================================

# 可视化统计滤波后的点云和噪声点云
o3d.visualization.draw_geometries([sor_pcd, sor_noise_pcd])