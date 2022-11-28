import time

import open3d as o3d
import numpy as np
y_threshold = 0.1
txt_path = '../txtcouldpoint/Finalzhengzheng5.txt'
# txt_path = 'txtcouldpoint/Original/Third_146.txt'
# txt_path = 'heidian.txt'


start_time = time.time()
# 通过numpy读取txt点云
pcd = np.loadtxt(txt_path, delimiter=",")
new_list=np.zeros([pcd.shape[0],6])
print(new_list.shape)
new_list[:,:3]=pcd

pcd_vector = o3d.geometry.PointCloud()
print(new_list[:, :6])

# 加载点坐标

pcd_vector.points = o3d.utility.Vector3dVector(new_list[:,:3])

# pcd_vector.paint_uniform_color([1,0.706,0])

#pcd_vector = pcd_vector.select_by_index(np.where(pcd[:, 2] <= y_threshold)[0])

end_time = time.time()
print(end_time-start_time)
o3d.visualization.draw_geometries([pcd_vector])