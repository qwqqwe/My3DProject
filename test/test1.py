# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证
import open3d as o3d
import os
import numpy as np
from pyntcloud import PyntCloud
from pandas import DataFrame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# matplotlib显示点云函数
def Point_Cloud_Show(points):
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], cmap='spectral', s=2, linewidths=0, alpha=1, marker=".")
    plt.title('Point Cloud')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


# 二维点云显示函数
def Point_Show(pca_point_cloud):
    x = []
    y = []
    pca_point_cloud = np.asarray(pca_point_cloud)
    for i in range(10000):
        x.append(pca_point_cloud[i][0])
        y.append(pca_point_cloud[i][1])
    plt.scatter(x, y)
    plt.show()


# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    # data => (10000, 3)  data_mean => (1, 3)
    data_mean = np.mean(data, axis=0)  # 对列求均值
    # normalize_data => (10000, 3)
    normalize_data = data - data_mean  # 数据归一化操作
    # H => (3, 3)
    H = np.dot(normalize_data.transpose(), normalize_data)
    # eigenvectors => (3,3)  eigenvalues => (3,)  eigenvectors_transpose => (3,3)
    eigenvectors, eigenvalues, eigenvectors_transpose = np.linalg.svd(H)  # SVD分解
    # 将特征值从大到小进行排序，便于提取主成分向量
    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors, normalize_data


def main():
    # 指定点云路径
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云

    # # 原始点云显示、可视化点云PCA之后的结果、PCA降维之后依据成分向量还原点云、用PCA分析点云主方向
    # *********************************************************************************
    # 1、加载原始点云(text)
    raw_point_cloud_matrix = np.genfromtxt(r"modelnet40_normal_resampled\airplane\airplane_0002.txt", delimiter=",")
    # raw_point_cloud_matrix_part = > (10000, 3)
    raw_point_cloud_matrix_part = raw_point_cloud_matrix[:, 0:3]
    raw_point_cloud = DataFrame(raw_point_cloud_matrix[:, 0:3])  # 选取每一列的第0至第2个元素
    raw_point_cloud.columns = ['x', 'y', 'z']
    point_cloud_pynt = PyntCloud(raw_point_cloud)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    o3d.visualization.draw_geometries([point_cloud_o3d])
    Point_Cloud_Show(raw_point_cloud_matrix_part)
    # 2、可视化点云PCA之后的结果
    eigenvalues, eigenvectors, normalize_data = PCA(raw_point_cloud_matrix_part)
    # vector => (3,2)
    vector = np.mat(eigenvectors[:, 0:2])
    # vector_transpose => (2,3)
    vector_transpose = vector.transpose()
    # pca_point_cloud_1 => (10000, 2)
    pca_point_cloud_1 = np.dot(normalize_data, vector)
    print(pca_point_cloud_1)
    # 3、PCA降维之后成分还原显示
    Point_Show(pca_point_cloud_1)
    # pca_point_cloud_1 => (10000, 3)
    pca_point_cloud_2 = np.dot(pca_point_cloud_1, vector_transpose)
    Point_Cloud_Show(pca_point_cloud_2)
    # 4、用PCA分析点云主方向
    primary_orientation_ = eigenvectors[:, 0]
    second_orientation = eigenvectors[:, 1]
    print('the main orientation of this pointcloud is: ', primary_orientation_)
    print('the second orientation of this pointcloud is: ', second_orientation)
    point = [[0, 0, 0], primary_orientation_, second_orientation]
    lines = [[0, 1], [0, 2]]
    colors = [[1, 0, 0], [0, 1, 0]]
    # 构造open3d中的LineSet对象，用于主成分和次主成分显示
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point), lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud_o3d, line_set])

    # 循环计算每个点的法向量
    # *********************************************************************************
    # 从点云中获取点，只对点进行处理
    points = point_cloud_pynt.points
    print('total points number is:', points.shape[0])
    normals = []
    # 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    # 每一点的法向量计算，通过PCA降维，对应最小特征值的成分向量近似为法向量
    for i in range(points.shape[0]):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[i], 20)
        k_nearest_point = np.asarray(point_cloud_o3d.points)[idx, :]
        w, v, _ = PCA(k_nearest_point)
        normals.append(v[:, 2])

    normals = np.array(normals, dtype=np.float64)
    print(normals.shape)
    # TODO: 此处把法向量存放在了normals中
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    # 法向量可视化，根据open3d文档，需要在显示窗口按住键‘n’才可以看到法向量
    o3d.visualization.draw_geometries([point_cloud_o3d])


if __name__ == '__main__':
    main()