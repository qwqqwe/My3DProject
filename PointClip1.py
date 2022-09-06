import open3d as o3d
import numpy as np


def plane_param(point_1, point_2, point_3):
    """
    不共线的三个点确定一个平面
    :param point_1: 点1
    :param point_2: 点2
    :param point_3: 点3
    :return: 平面方程系数:a,b,c,d
    """
    p1p2 = point_2 - point_1
    p1p3 = point_3 - point_1
    n = np.cross(p1p2, p1p3)  # 计算法向量

    n1 = n / np.linalg.norm(n)  # 单位法向量
    print(n1)
    A = n1[0]
    B = n1[1]
    C = n1[2]
    D = -A * point_1[0] - B * point_1[1] - C * point_1[2]
    return A, B, C, D


def visualizer_cloud(filtered):
    # ------------------------显示点云切片---------------------------
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='点云切片', width=800, height=600)
    # -----------------------可视化参数设置--------------------------
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])  # 设置背景色*****
    opt.point_size = 1  # 设置点的大小*************
    vis.add_geometry(filtered)  # 加载点云到可视化窗口
    vis.run()  # 激活显示窗口，这个函数将阻塞当前线程，直到窗口关闭。
    vis.destroy_window()  # 销毁窗口，这个函数必须从主线程调用。


if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud('drill/rabbit.pcd')
    point = np.asarray(pcd.points)
    # 1.获取位于点云上的三个点P1,P2,P3
    P1 = np.array([-0.089717, 0.118919, 0.045746])
    P2 = np.array([0.026236, 0.122255, 0.022206])
    P3 = np.array([0.058940, 0.056647, 0.020176])
    # 2.计算P1,P2,P3三点确定的平面，以此作为切片
    a, b, c, d = plane_param(P1, P2, P3)
    point_size = point.shape[0]
    idx = []
    # 3.设置切片厚度阈值，此值为切片厚度的一半
    Delta = 0.0005
    # 4.循环迭代查找满足切片的点
    for i in range(point_size):
        Wr = a * point[i][0] + b * point[i][1] + c * point[i][2] + d - Delta
        Wl = a * point[i][0] + b * point[i][1] + c * point[i][2] + d + Delta
        if Wr * Wl <= 0:
            idx.append(i)
    # 5.提取切片点云
    slicing_cloud = (pcd.select_by_index(idx))
    # 6.可视化切片
    visualizer_cloud(slicing_cloud)
