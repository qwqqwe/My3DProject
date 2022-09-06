import open3d as o3d

print("->正在加载点云... ")
pcd = o3d.io.read_point_cloud("../drill/Box.pcd")
print(pcd)

print("->正在RANSAC平面分割...")
distance_threshold = 0.2    # 内点到平面模型的最大距离
ransac_n = 3                # 用于拟合平面的采样点数
num_iterations = 1000       # 最大迭代次数

# 返回模型系数plane_model和内点索引inliers，并赋值
plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
print(inliers)
# 输出平面方程
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# 平面内点点云
inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([0, 0, 1.0])
print(inlier_cloud)

# 平面外点点云
outlier_cloud = pcd.select_by_index(inliers, invert=True)
outlier_cloud.paint_uniform_color([1.0, 0, 0])
print(outlier_cloud)

# 可视化平面分割结果
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])