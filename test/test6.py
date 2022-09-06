import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

print("->正在加载点云... ")
txt_path = '../txtcouldpoint/Depth_L5000_t220802_104306_01111.txt'
# 通过numpy读取txt点云
pcd_1 = np.genfromtxt(txt_path, delimiter=",")
pcd = o3d.geometry.PointCloud()
print(pcd_1.shape)

# 加载点坐标
pcd.points = o3d.utility.Vector3dVector(pcd_1[:, :3])

#pcd_vector = pcd_vector.select_by_index(np.where(pcd[:, 2] <= y_threshold)[0])

o3d.visualization.draw_geometries([pcd])
print(pcd)

print("->正在DBSCAN聚类...")
eps = 0.5           # 同一聚类中最大点间距
min_points = 50     # 有效聚类的最小点数
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd.cluster_dbscan(eps, min_points, print_progress=True))
max_label = labels.max()    # 获取聚类标签的最大值 [-1,0,1,2,...,max_label]，label = -1 为噪声，因此总聚类个数为 max_label + 1
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0  # labels = -1 的簇为噪声，以黑色显示
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd])