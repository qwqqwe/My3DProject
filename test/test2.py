import copy  # 点云深拷贝
import open3d as o3d
import numpy as np
def PCA(data, correlation=False, sort=True):
    # normalize 归一化
    mean_data = np.mean(data, axis=0)
    normal_data = data - mean_data
    # 计算对称的协方差矩阵
    H = np.dot(normal_data.T, normal_data)
    # SVD奇异值分解，得到H矩阵的特征值和特征向量
    eigenvectors, eigenvalues, _ = np.linalg.svd(H)

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors

# -------------------------- 加载点云 ------------------------
print("->正在加载点云... ")
pcd = o3d.io.read_point_cloud("../drill/Cylinder.pcd")
print(pcd)
pcd.paint_uniform_color([0,1,0])
print("->pcd质心：",pcd.get_center())

# ===========================================================

# -------------------------- 点云旋转 ------------------------
print("\n->采用欧拉角进行点云旋转")
pcd_EulerAngle = copy.deepcopy(pcd)
R1 = pcd.get_rotation_matrix_from_xyz((0,np.pi/2,0))
print("旋转矩阵：\n",R1)
pcd_EulerAngle.rotate(R1)    # 不指定旋转中心
pcd_EulerAngle.paint_uniform_color([0,0,1])
print("\n->pcd_EulerAngle质心：",pcd_EulerAngle.get_center())
# ===========================================================
points = np.asarray(pcd_EulerAngle.points)
# 用PCA分析点云主方向
w, v = PCA(points)  # PCA方法得到对应的特征值和特征向量
point_cloud_vector = v[:, 0]  # 点云主方向对应的向量为最大特征值对应的特征向量
print('the main orientation of this pointcloud is: ', point_cloud_vector)
print('v', v)
# 三个特征向量组成了三个坐标轴
axis = o3d.geometry.TriangleMesh.create_coordinate_frame().rotate(v, center=(0, 0, 0))
pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
# # 可视化
o3d.visualization.draw_geometries([pc_view, axis], point_show_normal=True)
#point = np.asarray(pcd.points)





# -------------------------- 可视化 --------------------------
#o3d.visualization.draw_geometries([pcd, pcd_EulerAngle])
# ===========================================================