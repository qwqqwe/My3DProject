import open3d as o3d
pcd = o3d.io.read_point_cloud("C:/Users/Administrator/PycharmProjects/My3DProject/drill/12.ply")
o3d.io.write_point_cloud("drill/sink_pointcloud11.pcd", pcd)