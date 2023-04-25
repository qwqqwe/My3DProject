import open3d as o3d
import numpy as np
import mayavi.mlab as mlab

# 4. 法向量的计算
def open3d_vector_compute():
    afile = 'fanzheng1'
    txt_path = '../../txtcouldpoint/Final{}.txt'.format(afile)

    defect_meassage = []
    # 通过numpy读取txt点云
    pcd_1 = np.loadtxt(txt_path, delimiter=",")
    pcd = o3d.geometry.PointCloud()
    # 加载点坐标
    pcd.points = o3d.utility.Vector3dVector(pcd_1)

    pcd = pcd.uniform_down_sample(50)


    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        # o3d.geometry.KDTreeSearchParamKNN(knn=20)  # 计算近邻的20个点
        # o3d.geometry.KDTreeSearchParamRadius(radius=0.01)  # 计算指定半径内的点
        # o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=20)  # 同时考虑搜索半径和近邻点个数

    )
    normals = np.array(pcd.normals)    # 法向量结果与点云维度一致(N, 3)
    points = np.array(pcd.points)
    print(normals.shape, points.shape)



    # 验证法向量模长为1(模长会有一定的偏差，不完全为1)
    normals_length = np.sum(normals**2, axis=1)
    flag = np.equal(np.ones(normals_length.shape, dtype=float), normals_length).all()
    print('all equal 1:', flag)

    # 法向量可视化
    o3d.visualization.draw_geometries([pcd],
                                         window_name="Open3d",
                                         # point_show_normal=True,
                                         width=800,   # 窗口宽度
                                         height=600)  # 窗口高度
def test():
    afile = 'fanzheng1'
    txt_path = '../../txtcouldpoint/Final{}.txt'.format(afile)

    defect_meassage = []
    # 通过numpy读取txt点云
    pcd_1 = np.loadtxt(txt_path, delimiter=",")
    pcd = o3d.geometry.PointCloud()
    # 加载点坐标
    pcd.points = o3d.utility.Vector3dVector(pcd_1)

    pcd = pcd.uniform_down_sample(200)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        # o3d.geometry.KDTreeSearchParamKNN(knn=20)  # 计算近邻的20个点
        # o3d.geometry.KDTreeSearchParamRadius(radius=0.01)  # 计算指定半径内的点
        # o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=20)  # 同时考虑搜索半径和近邻点个数

    )
    pcd.orient_normals_towards_camera_location(camera_location=[0., 0., 0.])
    pcd.normalize_normals()
    pcd.compute_point_cloud_distance()
    pcd.estimate_point_covariance()
    pcd.estimate_normals()


def compute_curvature(points, k=20):
    """
    计算点云的曲率
    Parameters
    ----------
    points : numpy.ndarray
        点云数组，形状为(n,3)，n为点的数量，每个点包含三个坐标值
    k : int
        用于计算曲率的邻居点数量
    Returns
    -------
    curvature : numpy.ndarray
        点云曲率数组，形状为(n,)
    """
    afile = 'fanzheng1'
    txt_path = '../../txtcouldpoint/Final{}.txt'.format(afile)

    defect_meassage = []
    # 通过numpy读取txt点云
    pcd_1 = np.loadtxt(txt_path, delimiter=",")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_1)
    # pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.uniform_down_sample(50)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    curvature = []
    for i, p in enumerate(pcd_1):
    # for i, p in enumerate(points):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(p, k)
        if k < 3:
            curvature.append(0)
        else:
            # cov = np.cov(points[idx].T)
            cov = np.cov(pcd_1[idx].T)
            eigvals = np.linalg.eigvalsh(cov)
            curvature.append(eigvals[0] / (eigvals[0] + eigvals[1] + eigvals[2]))
            # gaussian_curvature.append(eigvals[0] * eigvals[1] * eigvals[2] / np.sum(eigvals))
            # mean_curvature.append(0.5 * np.sum(eigvals) / k)

    print(curvature)
    np.savetxt("curvature.txt", curvature)
    return curvature

if __name__ == '__main__':
    # open3d_vector_compute()
    compute_curvature(1, k=20)

# import open3d as o3d
# import numpy as np
# def compute_curvature(points, k=20):
#     """
#     计算点云的曲率
#     Parameters
#     ----------
#     points : numpy.ndarray
#         点云数组，形状为(n,3)，n为点的数量，每个点包含三个坐标值
#     k : int
#         用于计算曲率的邻居点数量
#     Returns
#     -------
#     curvature : numpy.ndarray
#         点云曲率数组，形状为(n,)
#     """
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd_tree = o3d.geometry.KDTreeFlann(pcd)
#     curvature = []
#     for i, p in enumerate(points):
#         [k, idx, _] = pcd_tree.search_knn_vector_3d(p, k)
#         if k < 3:
#             curvature.append(0)
#         else:
#             cov = np.cov(points[idx].T)
#             eigvals = np.linalg.eigvalsh(cov)
#             curvature.append(eigvals[0] / (eigvals[0] + eigvals[1] + eigvals[2]))
#     return curvature
# 该程序使用Open3D中的geometry.PointCloud和geometry.KDTreeFlann类来计算点云中每个点的邻居点，
# 并使用Numpy库中的numpy.cov函数计算邻域内的协方差矩阵。然后，根据特征值计算点的曲率

# import open3d as o3d
# import numpy as np
# def compute_gaussian_curvature(points, k=20):
#     """
#     计算点云的高斯曲率
#     Parameters
#     ----------
#     points : numpy.ndarray
#         点云数组，形状为(n,3)，n为点的数量，每个点包含三个坐标值
#     k : int
#         用于计算曲率的邻居点数量
#     Returns
#     -------
#     gaussian_curvature : numpy.ndarray
#         点云高斯曲率数组，形状为(n,)
#     """
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd_tree = o3d.geometry.KDTreeFlann(pcd)
#     gaussian_curvature = []
#     for i, p in enumerate(points):
#         [k, idx, _] = pcd_tree.search_knn_vector_3d(p, k)
#         if k < 3:
#             gaussian_curvature.append(0)
#         else:
#             cov = np.cov(points[idx].T)
#             eigvals = np.linalg.eigvalsh(cov)
#             gaussian_curvature.append(eigvals[0] * eigvals[1] * eigvals[2] / np.sum(eigvals))
#     return gaussian_curvature
# 该程序与计算点云曲率的程序类似，只不过在计算曲率时使用了点云的高斯曲率公式，
# 即高斯曲率等于点云邻域内特征值的乘积除以特征值的和。

# import open3d as o3d
# import numpy as np
# def compute_mean_curvature(points, k=20):
#     """
#     计算点云的平均曲率
#     Parameters
#     ----------
#     points : numpy.ndarray
#         点云数组，形状为(n,3)，n为点的数量，每个点包含三个坐标值
#     k : int
#         用于计算曲率的邻居点数量
#     Returns
#     -------
#     mean_curvature : numpy.ndarray
#         点云平均曲率数组，形状为(n,)
#     """
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd_tree = o3d.geometry.KDTreeFlann(pcd)
#     mean_curvature = []
#     for i, p in enumerate(points):
#         [k, idx, _] = pcd_tree.search_knn_vector_3d(p, k)
#         if k < 3:
#             mean_curvature.append(0)
#         else:
#             cov = np.cov(points[idx].T)
#             eigvals = np.linalg.eigvalsh(cov)
#             mean_curvature.append(0.5 * np.sum(eigvals) / k)
#     return mean_curvature
# 该程序与计算点云曲率和高斯曲率的程序类似，只不过在计算曲率时使用了点云的平均曲率公式，
# 即平均曲率等于邻域内特征值之和的一半除以邻域大小