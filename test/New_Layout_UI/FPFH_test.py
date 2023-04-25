import open3d as o3d
import numpy as np
import time
from test_statistics import *
# import sklearn
from sklearn.decomposition import PCA

def compute_FPFH():
    afile = 'fanzheng1'
    txt_path = '../../txtcouldpoint/Final{}.txt'.format(afile)

    # 通过numpy读取txt点云
    pcd_1 = np.loadtxt(txt_path, delimiter=",")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_1)
    pcd = pcd.uniform_down_sample(50)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=20))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamKNN(knn=20))
    for i in range(pcd_fpfh.data.shape[0]):
        sum = 0
        for j in range(pcd_fpfh.data.shape[1]):
            sum += pcd_fpfh.data[i][j]
        print(i,sum/pcd_fpfh.data.shape[1])
    #0  10  15  17  28   十左右
    #5 16 27 150~175

    return pcd_fpfh


def compute_FPFH1(pd):


    # 通过numpy读取txt点云
    pcd_point = pd[:][:, 0:3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_point)


    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=20))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamKNN(knn=20))
    mean_list = []
    for i in range(pcd_fpfh.data.shape[0]):
        mean_list.append(np.mean(pcd_fpfh.data[i][:]))
    # for i in range(pcd_fpfh.data.shape[0]):
    #
    #     # for j in range(pcd_fpfh.data.shape[1]):
    #     #     sum += pcd_fpfh.data[i][j]
    #     sum = np.sum(pcd_fpfh.data[i][:])
    #     std = np.std(pcd_fpfh.data[i][:])
    #     var = np.var(pcd_fpfh.data[i][:])
    #     print(i,"mean:", sum / pcd_fpfh.data.shape[1],"std:",std,"var:",var)
        # print("std:",i,std)
    # 0 mean: 33.27009896662812 std: 28.294426117752895 var: 800.5745493329772
    # 1 mean: 0.005536968049438264 std: 0.19722580008342744 var: 0.03889801621854809
    # 2 mean: 0.08051108726262991 std: 1.118277862725387 var: 1.2505453782616596
    # 3 mean: 0.022898159996665505 std: 0.408904391788707 var: 0.16720280162409243
    # 4 mean: 0.074597738154821 std: 1.0900150815181533 var: 1.1881328779370262
    # 5 mean: 133.4954283505148 std: 42.72241562372648 var: 1825.2047967264282
    # 6 mean: 0.4035031890008413 std: 2.462375925548506 var: 6.063295198720862
    # 7 mean: 0.029248401092147994 std: 0.6301056315830806 var: 0.39703310695271293
    # 8 mean: 0.06398330307224626 std: 0.7611861081877593 var: 0.5794042912980273
    # 9 mean: 0.016504794124351407 std: 0.4295214805205943 var: 0.18448870222860328
    # 10 mean: 32.53768904210397 std: 28.573543191165115 var: 816.4473704973783
    # 11 mean: 0.9321296542234532 std: 6.609092401219332 var: 43.68010236785512
    # 12 mean: 0.01388337890667996 std: 0.2619666556702076 var: 0.06862652868303312
    # 13 mean: 0.01250478805640148 std: 0.243023948008757 var: 0.05906063930576303
    # 14 mean: 0.04241678847085583 std: 0.7118032702338938 var: 0.5066638955156656
    # 15 mean: 11.841249656825116 std: 12.54442698245761 var: 157.36264831821052
    # 16 mean: 174.32665412366265 std: 23.69451115789332 var: 561.429859011531
    # 17 mean: 11.829196528529572 std: 12.393372605180529 var: 153.59568453083924
    # 18 mean: 0.05760373445472051 std: 1.1602606181399158 var: 1.3462047020064196
    # 19 mean: 0.013036919553425903 std: 0.2519684384180626 var: 0.06348809395883702
    # 20 mean: 0.018236354852926427 std: 0.3102757621394828 var: 0.09627104857123693
    # 21 mean: 0.9130880724642638 std: 6.525641823631935 var: 42.584001210334335
    # 22 mean: 0.8815998645692436 std: 2.85312741372926 var: 8.140336038973416
    # 23 mean: 0.61969715567838 std: 2.1264265874165686 var: 4.521690031672074
    # 24 mean: 0.980456788786706 std: 2.796956371784881 var: 7.822964945668045
    # 25 mean: 2.3891669908157054 std: 4.94520637649159 var: 24.45506610609308
    # 26 mean: 36.455514698763594 std: 25.689853754536244 var: 659.96858592946
    # 27 mean: 135.6422097497747 std: 34.33579003174527 var: 1178.9464771040978
    # 28 mean: 19.033304347529413 std: 21.4189882054628 var: 458.77305574575445
    # 29 mean: 1.7771123166419174 std: 4.396313264770254 var: 19.32757032199489
    # 30 mean: 0.7901727382202197 std: 2.4126353209672913 var: 5.820809191978945
    # 31 mean: 0.5942032240413558 std: 2.115359823682091 var: 4.474747183648328
    # 32 mean: 0.8365621251787927 std: 2.6291912860627686 var: 6.912646818708395
    list=[]

    sizep = pd.shape[0]


    for j in range(pcd_fpfh.data.shape[1]):
        if pd[j][3] == 0:
            list.append(pcd_fpfh.data[:, j])


    list1 = np.asarray(list)
    print(list1.shape)
    # for i in range(list1.shape[0]):
    #     # for j in range(pcd_fpfh.data.shape[1]):
    #     #     sum += pcd_fpfh.data[i][j]
    #     mean = np.mean(list1[i][:])
    #     std = np.std(list1[i][:])
    #     var = np.var(list1[i][:])
    #     print(i, "mean:", mean, "std:", std, "var:", var)
    #0 mean: 0.013036919553425903 std: 0.2519684384180626 var: 0.06348809395883702
    # 1 mean: 0.018236354852926427 std: 0.3102757621394828 var: 0.09627104857123693
    # 2 mean: 0.9130880724642638 std: 6.525641823631935 var: 42.584001210334335
    # 3 mean: 0.8815998645692436 std: 2.85312741372926 var: 8.140336038973416
    # 4 mean: 0.61969715567838 std: 2.1264265874165686 var: 4.521690031672074
    # 5 mean: 0.980456788786706 std: 2.796956371784881 var: 7.822964945668045
    # 6 mean: 2.3891669908157054 std: 4.94520637649159 var: 24.45506610609308
    # 7 mean: 36.455514698763594 std: 25.689853754536244 var: 659.96858592946
    # 8 mean: 135.6422097497747 std: 34.33579003174527 var: 1178.9464771040978
    # 9 mean: 19.033304347529413 std: 21.4189882054628 var: 458.77305574575445
    # 10 mean: 1.7771123166419174 std: 4.396313264770254 var: 19.32757032199489
    # 11 mean: 0.7901727382202197 std: 2.4126353209672913 var: 5.820809191978945
    # 12 mean: 0.5942032240413558 std: 2.115359823682091 var: 4.474747183648328
    # 13 mean: 0.8365621251787927 std: 2.6291912860627686 var: 6.912646818708395

    print("123")

    # list = []
    # for i in range (pcd_fpfh.data.shape[1]):
    #     if pd[i][3]==0:
    #         list.append(pcd_fpfh.data[5][i])
    #         list.append(pcd_fpfh.data[16][i])
    #         list.append(pcd_fpfh.data[27][i])
    #         list.append(pcd_fpfh.data[0][i])
    #
    # print("1")


    return list1,mean_list

# def PCA(data, correlation=False, sort=True):
#     # normalize 归一化
#     mean_data = np.mean(data, axis=0)
#     normal_data = data - mean_data
#     # 计算对称的协方差矩阵
#     H = np.dot(normal_data.T, normal_data)
#     # SVD奇异值分解，得到H矩阵的特征值和特征向量
#     eigenvectors, eigenvalues, _ = np.linalg.svd(H)
#
#     if sort:
#         sort = eigenvalues.argsort()[::-1]    #对特征向量进行排序，从大到小，返回索引值
#         eigenvalues = eigenvalues[sort]
#         eigenvectors = eigenvectors[:, sort]
#
#     return eigenvalues, eigenvectors
# 0	[9.48633317e+01]
# 1	 [-5.52791358e+02]
# 2	 [-5.52247671e+02]
# 3	 [-5.53166531e+02]
# 4	 [-5.31129734e+02]
# 5	 [ 4.20781649e+03]
# 6	 [-4.58827707e+02]
# 7	 [-5.53166670e+02]
# 8	 [-5.51887819e+02]
# 9	 [-5.53055139e+02]
# 10	 [ 3.59280598e+00]
# 11	 [-5.28215561e+02]
# 12	 [-5.53166335e+02]
# 13	 [-5.52973105e+02]
# 14	 [-5.50079425e+02]
# 15	 [-2.86007972e+01]
# 16	 [ 4.44778371e+03]
# 17	 [-5.17865603e+01]
# 18	 [-5.49970411e+02]
# 19	 [-5.53059954e+02]
# 20	 [-5.52765986e+02]
# 21	 [-5.27165570e+02]
# 22	 [-5.34784256e+02]
# 23	 [-5.35044601e+02]
# 24	 [-5.33524742e+02]
# 25	 [-4.86648552e+02]
# 26	 [ 7.63168380e+02]
# 27	 [ 2.94057008e+03]
# 28	 [ 4.89155135e+02]
# 29	 [-5.02072096e+02]
# 30	 [-5.27278662e+02]
# 31	 [-5.37222625e+02]
# 32	 [-5.36318063e+02]

if __name__ == '__main__':

    pd, _ = display2()
    pcd_point = pd[:][:, 0:3]
    d = pd[:][:, 0:3]

    st = time.time()

    data,mean_list = compute_FPFH1(pd)
    data1 = data.T
    # compute_FPFH()
    # n_components 指明了降到几维
    pca = PCA(n_components=1)

    # 利用数据训练模型（即上述得出特征向量的过程）
    pca.fit(data1)

    # 得出原始数据的降维后的结果；也可以以新的数据作为参数，得到降维结果。
    print(pca.transform(data1))
    # todo:降维后的特征向量与原始的均值乘积

    # 打印各主成分的方差占比
    print(pca.explained_variance_ratio_)

    for i in range(0,33):
        print(i,pca.transform(data1)[i]*mean_list[i])

    # 0[3156.11243426]
    # 1[-3.06078809]
    # 2[-44.46206046]
    # 3[-12.66649573]
    # 4[-39.62107683]
    # 5[561724.26502963]
    # 6[-185.13844288]
    # 7[-16.17924063]
    # 8[-35.31160561]
    # 9[-9.12806122]
    # 10[116.90160373]
    # 11[-492.36538811]
    # 12[-7.67981783]
    # 13[-6.91481148]
    # 14[-23.3326026]
    # 15[-338.66917953]
    # 16[775367.25162802]
    # 17[-612.59339924]
    # 18[-31.68034951]
    # 19[-7.21019813]
    # 20[-10.08043668]
    # 21[-481.34859457]
    # 22[-471.46572734]
    # 23[-331.56561734]
    # 24 [-523.09795522]
    # 25[-1162.68465685]
    # 26[27821.69610848]
    # 27[398865.42370009]
    # 28[9310.23855396]
    # 29[-892.23850511]
    # 30[-416.641224]
    # 31[-319.21941607]
    # 32[-448.66337819]

    et = time.time()
    print("time:",et-st)
    # print(et-st)
    print("1")
