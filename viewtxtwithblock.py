import time

import open3d as o3d
import numpy as np
y_threshold = 0.1
txt_path = 'txtcouldpoint/Finaldistroy1.txt'
# txt_path = 'txtcouldpoint/Original/Third_146.txt'

start_time = time.time()
# 通过numpy读取txt点云
pcd = np.genfromtxt(txt_path, delimiter=" ")
pcd_vector = o3d.geometry.PointCloud()
print(pcd.shape)

# 加载点坐标
pcd_vector.points = o3d.utility.Vector3dVector(pcd[:, :3])

#pcd_vector = pcd_vector.select_by_index(np.where(pcd[:, 2] <= y_threshold)[0])

end_time = time.time()
print(end_time-start_time)
o3d.visualization.draw_geometries([pcd_vector])