import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

#点云加载

print("->正在加载点云... ")
pcd = o3d.io.read_point_cloud("drill/Box.pcd")

#点云统计滤波

print("->正在进行统计滤波...")
num_neighbors = 20 # K邻域点的个数
std_ratio = 2.0 # 标准差乘数
# 执行统计滤波，返回滤波后的点云sor_pcd和对应的索引ind
sor_pcd, ind = pcd.remove_statistical_outlier(num_neighbors, std_ratio)
sor_pcd.paint_uniform_color([0, 0, 1])
print("统计滤波后的点云：", sor_pcd)
sor_pcd.paint_uniform_color([0, 0, 1])
# 提取噪声点云
sor_noise_pcd = pcd.select_by_index(ind,invert = True)
print("噪声点云：", sor_noise_pcd)
sor_noise_pcd.paint_uniform_color([1, 0, 0])

#点云获取二维化

pdd=np.asarray(pcd.points)
pcc=pdd[:, 0:2]
x_label=pcc[0:,0]
y_label=pcc[0:,1]
print(x_label.max(),x_label.min())
print(y_label.max(),y_label.min())
plt.scatter(x_label,y_label,marker='o')

plt.xlabel('1st')
plt.ylabel('2nd')
plt.title('Scatter Plot')
plt.show()
#








# print("->正在DBSCAN聚类...")
# eps = 0.5           # 同一聚类中最大点间距
# min_points = 50     # 有效聚类的最小点数
# with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
#     labels = np.array(pcd.cluster_dbscan(eps, min_points, print_progress=True))
# max_label = labels.max()    # 获取聚类标签的最大值 [-1,0,1,2,...,max_label]，label = -1 为噪声，因此总聚类个数为 max_label + 1
# print(f"point cloud has {max_label + 1} clusters")
# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# colors[labels < 0] = 0  # labels = -1 的簇为噪声，以黑色显示
# pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([pcd])