
import open3d as o3d
import numpy as np
from numpy.linalg import norm
from numpy.linalg import svd
from line_profiler import LineProfiler
from functools import wraps
import time
# import heartrate
from numba import jit
from test_statistics import *



#整个的高斯曲率

def planeFit(points):
    """
    p, n = planeFit(points)

    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    """
    points = np.transpose(points)
    points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:, np.newaxis]
    M = np.dot(x, x.T) # Could also use np.cov(x) here.
    return ctr, svd(M)[0][:, -1]

def func_line_time(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        func_return = f(*args, **kwargs)
        lp = LineProfiler()
        lp_wrap = lp(f)
        lp_wrap(*args, **kwargs)
        lp.print_stats()
        return func_return

    return decorator

# @func_line_time
def quadric_equation(X, Y, Z):
    """
    Helper function used in compute_mean_curvature() and compute_gaussian_curvature() that fits a quadric surface to
    a list of points and returns the constants of the quadric equation of the surface. For more information, see the
    documentation on those functions or https://github.com/rbv188/quadric-curve-fit/blob/master/quadrics.py
    :param X: array of x coordinates
    :param Y: array of y coordinates
    :param Z: array of z coordinates
    :return: the 9 equation coefficients
    """
    X = X.reshape((len(X), 1))
    Y = Y.reshape((len(Y), 1))
    Z = Z.reshape((len(Z), 1))
    num = len(X)

    # matrix = np.hstack((X**2, Y**2, Z**2, 2*X*Y, 2*X*Z, 2*Y*Z, 2*X, 2*Y, 2*Z)) #45%花费时间
    matrix = np.hstack((X *X, Y * Y, Z * Z, 2 * X * Y, 2 * X * Z, 2 * Y * Z, 2 * X, 2 * Y, 2 * Z))
    output = np.ones((num, 1))

    [a, b, c, d, e, f, g, h, i] = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(matrix), matrix)),
                                                np.transpose(matrix)), output)

    a = -a[0]
    b = -b[0]
    c = -c[0]
    d = -d[0]
    e = -e[0]
    f = -f[0]
    g = -g[0]
    h = -h[0]
    i = -i[0]
    j = 1

    constants = np.array([a, b, c, d, e, f, g, h, i, j])
    return constants


from math import sqrt


def getclosestpoint(pt, consts):
    """
    fast closest point estimate on an implicit quadric surface using line intersections
    :param pt: an initial pt estimate
    :param consts: the coefficients defining the surface
    :return: a pt as a numpy array
    """
    lines = {'n1_itsc': None, 'n2_itsc': None, 'n3_itsc': None, 'n4_itsc': None}
    lines['n1_itsc'] = getitsc(consts, pt, 'x')
    lines['n2_itsc'] = getitsc(consts, pt, 'y')
    lines['n3_itsc'] = getitsc(consts, pt, 'z')
    if len([line for line in lines.values() if line is not None]) >= 2:
        direction = getaveragedirection(lines, pt)
        lines['n4_itsc'] = getitsc(consts, pt, 'avg', v=direction)

    if lines['n4_itsc'] is not None:
        return lines['n4_itsc']
    else:
        values = [line for line in lines.values() if line is not None]
        if len(values) < 1:
            return pt
        else:
            return values[0]

# @func_line_time
def getitsc(ct, pt, dim, v=None):
    """
    solve surface equation for single variable, algebra done ahead of time so all the computer has to do
    is crunch numbers directly.
    :param ct: list of constants for the surface equation
    :param pt: the estimated point close to the surface
    :param dim: which variable to solve for (x, y, z, or avg for n4)
    :param v: required when solving for n4, a direction vector
    :return: a point lying on the surface
    """
    # a b c e f g l m n d
    # 0 1 2 3 4 5 6 7 8 9
    # pt = [p1, p2, p3] in expressions, access with pt[0], pt[1], pt[2]
    # v = [v1 v2 v3] in expressions, access with v[0], v[1], v[2]
    if dim == 'x':
        a = ct[0]  # ax^2
        b = 2*ct[3]*pt[1] + 2*ct[5]*pt[2] + 2*ct[6]  # (2ep2 + 2gp3 +2l)x
        # c = ct[1]*pt[1]**2 + ct[2]*pt[2]**2 + 2*ct[4]*pt[1]*pt[2] + 2*ct[7]*pt[1] + 2*ct[8]*pt[2] + ct[9]  # bp2^2 + cp3^2 + 2fp2p3 + 2mp2 + 2np3 + d
        c = ct[1]*pt[1]*pt[1] + ct[2]*pt[2]*pt[2] + 2*ct[4]*pt[1]*pt[2] + 2*ct[7]*pt[1] + 2*ct[8]*pt[2] + ct[9]  # bp2^2 + cp3^2 + 2fp2p3 + 2mp2 + 2np3 + d
    elif dim == 'y':
        a = ct[1]  # by^2
        b = 2*ct[3]*pt[0] + 2*ct[4]*pt[2] + 2*ct[7]  # (2ep1 + 2fp3 +2m)y
        # c = ct[0]*pt[0]**2 + ct[2]*pt[2]**2 + 2*ct[5]*pt[2]*pt[0] + 2*ct[6]*pt[0] + 2*ct[8]*pt[2] + ct[9]  # ap1^2 + cp3^2 + 2gp1p3 + 2lp1 + 2np3 + d
        c = ct[0]*pt[0]*pt[0] + ct[2]*pt[2]*pt[2] + 2*ct[5]*pt[2]*pt[0] + 2*ct[6]*pt[0] + 2*ct[8]*pt[2] + ct[9]  # ap1^2 + cp3^2 + 2gp1p3 + 2lp1 + 2np3 + d
    elif dim == 'z':
        a = ct[2]  # cz^2
        b = 2*ct[4]*pt[1] + 2*ct[5]*pt[0] + 2*ct[8]  # (2fp2 + 2gp1 +2n)z
        # c = ct[0]*pt[0]**2 + ct[1]*pt[1]**2 + 2*ct[3]*pt[0]*pt[1] + 2*ct[6]*pt[0] + 2*ct[7]*pt[1] + ct[9]  # ap1^2 + bp2^2 + 2ep1p2 + 2lp1 + 2mp2 + d
        c = ct[0]*pt[0]*pt[0] + ct[1]*pt[1]*pt[1] + 2*ct[3]*pt[0]*pt[1] + 2*ct[6]*pt[0] + 2*ct[7]*pt[1] + ct[9]  # ap1^2 + bp2^2 + 2ep1p2 + 2lp1 + 2mp2 + d
    elif dim == 'avg':
        a = ct[0]*v[0]*v[0] + ct[1]*v[1]*v[1] + ct[2]*v[2]*v[2] + 2*ct[3]*v[0]*v[1] + 2*ct[4]*v[1]*v[2] + 2*ct[5]*v[0]*v[2]  # (av1^2 + bv2^2 + cv3^2 + 2ev1v2 + 2fv2v3 + 2gv1v3)t^2
        b = 2*ct[0]*pt[0]*v[0] + 2*ct[1]*pt[1]*v[1] + 2*ct[2]*pt[2]*v[2] + 2*ct[3]*pt[0]*v[1] + 2*ct[3]*pt[1]*v[0] \
        + 2*ct[4]*pt[1]*v[2] + 2*ct[4]*pt[2]*v[1] + 2*ct[5]*pt[2]*v[0] + 2*ct[5]*pt[0]*v[2] + 2*ct[6]*v[0] + 2*ct[7]*v[1] +2*ct[8]*v[2]  # (2ap1v1 + 2bp2v2 + 2cp3v3 + 2ep1v2 + 2ep2v1 + 2fp2v3 + 2fp3v2 + 2gp3v1 + 2gp1v3 + 2lv1 + 2mv2 + 2nv3)t
        c = ct[0]*pt[0]*pt[0] + ct[1]*pt[1]*pt[1] + ct[2]*pt[2]*pt[2] + 2*ct[3]*pt[0]*pt[1] + 2*ct[4]*pt[1]*pt[2] \
        + 2*ct[5]*pt[0]*pt[2] + 2*ct[6]*pt[0] + 2*ct[7]*pt[1] + 2*ct[8]*pt[2] + ct[9]  # ap1^2 + bp2^2 + cp3^2 + 2ep1p2 + 2fp2p3 + 2gp1p3 + 2lp1 + 2mp2 +2np3 +d
    else:
        raise Exception(f"{dim} is not a valid dim")

    # compute roots with quadratic equation
    roots = []
    itsc = None
    hasroots = False
    try:
        roots.append((-b + sqrt(b * b - 4 * a * c)) / (2 * a))
        hasroots = True
    except:
        pass
    try:
        roots.append((-b - sqrt(b * b - 4 * a * c)) / (2 * a))
        hasroots = True
    except:
        pass

    # handle solutions if they exist
    if hasroots:
        if dim == 'x':
            newpt1 = np.array([roots[0], pt[1], pt[2]])
        elif dim == 'y':
            newpt1 = np.array([pt[0], roots[0], pt[2]])
        elif dim == 'z':
            newpt1 = np.array([pt[0], pt[1], roots[0]])
        else:
            newpt1 = pt + roots[0] * v
        if len(roots) == 2:
            if dim == 'x':
                newpt2 = np.array([roots[1], pt[1], pt[2]])
            elif dim == 'y':
                newpt2 = np.array([pt[0], roots[1], pt[2]])
            elif dim == 'z':
                newpt2 = np.array([pt[0], pt[1], roots[1]])
            else:
                newpt2 = pt + roots[1] * v
            if np.linalg.norm(pt - newpt1) < np.linalg.norm(pt - newpt2):
                itsc = newpt1
            else:
                itsc = newpt2
        else:
            itsc = newpt1
    return itsc


def getaveragedirection(lines, pt):
    """
    simple vector average
    :param lines: a line represented as a point, implicitly originating from the pt argument
    :param pt: a point from which the lines originate
    :return: new direction vector
    """
    avg = np.zeros(3)
    n = 0
    for line in lines:
        if lines[line] is not None:
            vec = lines[line] - pt
            avg += vec
            n += 1
    avg /= n
    return avg

# @jit
# @func_line_time
def compute_fundamentals(point, constants):
    '''
    Helper function used in compute_mean_curvature() and compute_gaussian_curvature() that computes the k-values of
    curvature by finding the First and Second Fundamental Forms (E, F, G, L, M, N). For more information, see the
    documentation on those functions.
    :param point: the current point on the quadric surface
    :param constants: the constants of the quadric surface
    :return: the k-values of the curvature of the quadric surface at the point
    '''
    Fx = 2*constants[0]*point[0] + constants[3]*point[1] + constants[5]*point[2] + constants[6]
    Fy = 2*constants[1]*point[1] + constants[3]*point[0] + constants[4]*point[2] + constants[7]
    Fz = 2*constants[2]*point[2] + constants[4]*point[1] + constants[5]*point[0] + constants[8]
    Fxx = 2*constants[0]
    Fyy = 2*constants[1]
    Fzz = 2*constants[2]
    Fxy = constants[3]
    Fyz = constants[4]
    Fxz = constants[5]
    grad_F = sqrt(Fx *Fx + Fy *Fy + Fz *Fz)

    # from https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.98.7059&rep=rep1&type=pdf
    E = 1 + (Fx*Fx / Fz*Fz)
    F = Fx * Fy / Fz*Fz
    G = 1 + (Fy*Fy / Fz*Fz)
    L = (1 / (Fz*Fz * grad_F)) * np.linalg.det(np.array(([Fxx, Fxz, Fx], [Fxz, Fzz, Fz], [Fx, Fz, 0])))
    M = (1 / (Fz*Fz * grad_F)) * np.linalg.det(np.array(([Fxy, Fyz, Fy], [Fxz, Fzz, Fz], [Fx, Fz, 0])))
    N = (1 / (Fz*Fz * grad_F)) * np.linalg.det(np.array(([Fyy, Fyz, Fy], [Fyz, Fzz, Fz], [Fy, Fz, 0])))

    A = np.array(([L, M], [M, N]))
    B = np.array(([E, F], [F, G]))
    B_inv = np.linalg.inv(B)

    k_values = np.linalg.eigvals(np.dot(B_inv, A))
    return k_values


def compute_gaussian_curvature(pc, pcd_tree, PPEexec=0, radius=0.2, k=20):
    """

    过找到最佳拟合曲面的方程开始查找点的高斯曲率
    由周围的点创建，格式为（在方法二次方程（）中）：
    a*x**2+b*y**2+c*z**2+d*x*y+e*y*z+f*x*z+g*x+h*y+i*z+j=0
    其中（a，b，c，d，e，f，g，h，i，j）是常数。接下来，距离当前感兴趣点最近的点
    使用getclosestpoint（）方法（请参见surface_intersection.py）找到最佳拟合曲面上的点。然后
    程序找到方程关于x、y和z的一阶和二阶偏导数
    方程的梯度。最近点的坐标用于本表中x、y和z的值
    计算然后，程序使用以下公式计算高斯曲率（在该方法中
    compute_fundationals（））：
    E=1+（Fx**2/Fz**2）
    F=Fx*Fy/Fz**2
    G=1+（Fy**2/Fz**2）
    L=（1/（Fz**2*grad_F））*det（（[Fxx，Fxz，Fx]，[Fxz，Fzz，Fz]，[Fx，Fz，0]））
    M=（1/（Fz**2*grad_F））*det（（[Fxy，Fyz，Fy]，[Fxz，Fzz，Fz]，[Fx，Fz，0]））
    N=（1/（Fz**2*grad_F））*det（（[Fyy，Fyz，Fy]，[Fyz，Fzz，Fz]，[Fy，Fz，0]））
    A=（[L，M]，[M，N]）
    B=（[E，F]，[F，G]）
    B_inv=逆（B）
    k_values=特征值（点（B_inv，A））
    其中Fx是函数相对于x的一阶导数，Fy是函数的一阶微分
    关于y，Fz是函数关于z的一阶导数，Fxx是函数的二阶导数
    关于x的函数，Fyy是关于y的函数的二阶导数，Fzz是二阶导数
    函数对z的导数，Fxy是函数对x的二阶导数
    然后到y，Fyz是函数相对于y的二阶导数，然后到z，Fxz是二阶导数
    函数对x的导数，然后对z的导数，grad_F是
    作用因此，变量E、F和G是第一基本形式的系数
    五十、 M和N是第二基本形式的系数
    B_inv*A（两个矩阵的点积）的特征值，即k1和k2。
    最后，计算点的高斯曲率：
    高斯曲率=k1*k2
    为了在cloudcompare中更容易可视化，将生成的值转换为1是最小值
    然后取每个值的对数。

    """

    print("determining arg lists...")
    sizep = pc.shape[0]
    angle_list = []
    for pt_idx in range(sizep):
        # [kk, idx, _] = pcd_tree.search_hybrid_vector_3d(pc[pt_idx], radius, k)
        [kk, idx, _] = pcd_tree.search_knn_vector_3d(pc[pt_idx], k)

        if kk < 6:
            # return np.nan
            continue
        # get the plane of best fit as a point and normal vector  [idx[:], :]
        # pt, normal = planeFit(data[idx])

        heights = pc[idx]
        heights = heights[:, 2]
        # hmax = np.max(heights)
        # hmin = np.min(heights)

        curve_points = pc[idx]
        current_pt = pc[pt_idx]
        constants = quadric_equation(curve_points[:, 0], curve_points[:, 1], curve_points[:, 2])
        estimate_pt = getclosestpoint(current_pt, constants)
        k_values = compute_fundamentals(estimate_pt, constants)
        g_curvature = np.real(k_values[0] * k_values[1])  # K

        angle_list.append(abs(g_curvature * 100))


    return angle_list


# @func_line_time
def compute_gaussian_curvature1(pc, pcd_tree, pd, k=20):
    """
    过找到最佳拟合曲面的方程开始查找点的高斯曲率
    由周围的点创建，格式为（在方法二次方程（）中）：
    a*x**2+b*y**2+c*z**2+d*x*y+e*y*z+f*x*z+g*x+h*y+i*z+j=0
    其中（a，b，c，d，e，f，g，h，i，j）是常数。接下来，距离当前感兴趣点最近的点
    使用getclosestpoint（）方法（请参见surface_intersection.py）找到最佳拟合曲面上的点。然后
    程序找到方程关于x、y和z的一阶和二阶偏导数
    方程的梯度。最近点的坐标用于本表中x、y和z的值
    计算然后，程序使用以下公式计算高斯曲率（在该方法中
    compute_fundationals（））：
    E=1+（Fx**2/Fz**2）
    F=Fx*Fy/Fz**2
    G=1+（Fy**2/Fz**2）
    L=（1/（Fz**2*grad_F））*det（（[Fxx，Fxz，Fx]，[Fxz，Fzz，Fz]，[Fx，Fz，0]））
    M=（1/（Fz**2*grad_F））*det（（[Fxy，Fyz，Fy]，[Fxz，Fzz，Fz]，[Fx，Fz，0]））
    N=（1/（Fz**2*grad_F））*det（（[Fyy，Fyz，Fy]，[Fyz，Fzz，Fz]，[Fy，Fz，0]））
    A=（[L，M]，[M，N]）
    B=（[E，F]，[F，G]）
    B_inv=逆（B）
    k_values=特征值（点（B_inv，A））
    其中Fx是函数相对于x的一阶导数，Fy是函数的一阶微分
    关于y，Fz是函数关于z的一阶导数，Fxx是函数的二阶导数
    关于x的函数，Fyy是关于y的函数的二阶导数，Fzz是二阶导数
    函数对z的导数，Fxy是函数对x的二阶导数
    然后到y，Fyz是函数相对于y的二阶导数，然后到z，Fxz是二阶导数
    函数对x的导数，然后对z的导数，grad_F是
    作用因此，变量E、F和G是第一基本形式的系数
    五十、 M和N是第二基本形式的系数
    B_inv*A（两个矩阵的点积）的特征值，即k1和k2。
    最后，计算点的高斯曲率：
    高斯曲率=k1*k2
    为了在cloudcompare中更容易可视化，将生成的值转换为1是最小值
    然后取每个值的对数。

    """

    print("determining arg lists...")
    sizep = pc.shape[0]
    angle_list = []

    for pt_idx in range(sizep):
        # [kk, idx, _] = pcd_tree.search_hybrid_vector_3d(pc[pt_idx], radius, k)
        [kk, idx, _] = pcd_tree.search_knn_vector_3d(pc[pt_idx], k)

        if kk < 6:
            # return np.nan
            continue
        # get the plane of best fit as a point and normal vector  [idx[:], :]
        # pt, normal = planeFit(data[idx])
        if pd[pt_idx][3] != 1:
            # [kk, idx, _] = pcd_tree.search_knn_vector_3d(pc[pt_idx], k)
            # heights = pc[idx]
            # heights = heights[:, 2]
            # hmax = np.max(heights)
            # hmin = np.min(heights)
            curve_points = pc[idx]
            current_pt = pc[pt_idx]
            constants = quadric_equation(curve_points[:, 0], curve_points[:, 1], curve_points[:, 2])
            estimate_pt = getclosestpoint(current_pt, constants)
            k_values = compute_fundamentals(estimate_pt, constants)
            g_curvature = np.real(k_values[0] * k_values[1])  # K
            m_curvature = np.real(k_values[0]+k_values[1])/2
            cu_list = []
            cu_list.append(g_curvature)
            cu_list.append(m_curvature)
            angle_list.append(cu_list)

    return angle_list

# @jit
# @func_line_time
def compute_curvature1(pc, pcd_tree, pd, k=20):
    """
    过找到最佳拟合曲面的方程开始查找点的高斯曲率
    由周围的点创建，格式为（在方法二次方程（）中）：
    a*x**2+b*y**2+c*z**2+d*x*y+e*y*z+f*x*z+g*x+h*y+i*z+j=0
    其中（a，b，c，d，e，f，g，h，i，j）是常数。接下来，距离当前感兴趣点最近的点
    使用getclosestpoint（）方法（请参见surface_intersection.py）找到最佳拟合曲面上的点。然后
    程序找到方程关于x、y和z的一阶和二阶偏导数
    方程的梯度。最近点的坐标用于本表中x、y和z的值
    计算然后，程序使用以下公式计算高斯曲率（在该方法中
    compute_fundationals（））：
    E=1+（Fx**2/Fz**2）
    F=Fx*Fy/Fz**2
    G=1+（Fy**2/Fz**2）
    L=（1/（Fz**2*grad_F））*det（（[Fxx，Fxz，Fx]，[Fxz，Fzz，Fz]，[Fx，Fz，0]））
    M=（1/（Fz**2*grad_F））*det（（[Fxy，Fyz，Fy]，[Fxz，Fzz，Fz]，[Fx，Fz，0]））
    N=（1/（Fz**2*grad_F））*det（（[Fyy，Fyz，Fy]，[Fyz，Fzz，Fz]，[Fy，Fz，0]））
    A=（[L，M]，[M，N]）
    B=（[E，F]，[F，G]）
    B_inv=逆（B）
    k_values=特征值（点（B_inv，A））
    其中Fx是函数相对于x的一阶导数，Fy是函数的一阶微分
    关于y，Fz是函数关于z的一阶导数，Fxx是函数的二阶导数
    关于x的函数，Fyy是关于y的函数的二阶导数，Fzz是二阶导数
    函数对z的导数，Fxy是函数对x的二阶导数
    然后到y，Fyz是函数相对于y的二阶导数，然后到z，Fxz是二阶导数
    函数对x的导数，然后对z的导数，grad_F是
    作用因此，变量E、F和G是第一基本形式的系数
    五十、 M和N是第二基本形式的系数
    B_inv*A（两个矩阵的点积）的特征值，即k1和k2。
    最后，计算点的高斯曲率：
    高斯曲率=k1*k2
    为了在cloudcompare中更容易可视化，将生成的值转换为1是最小值
    然后取每个值的对数。

    """

    print("determining arg lists...")
    sizep = pc.shape[0]
    angle_list = []

    for pt_idx in range(sizep):
        # [kk, idx, _] = pcd_tree.search_hybrid_vector_3d(pc[pt_idx], radius, k)
        # [kk, idx, _] = pcd_tree.search_knn_vector_3d(pc[pt_idx], k)

        # if kk < 6:
        #     # return np.nan
        #     continue
        # get the plane of best fit as a point and normal vector  [idx[:], :]
        # pt, normal = planeFit(data[idx])
        if pd[pt_idx][3] != 1:
            [kk, idx, _] = pcd_tree.search_knn_vector_3d(pc[pt_idx], k)
            # heights = pc[idx]
            # heights = heights[:, 2]
            # hmax = np.max(heights)
            # hmin = np.min(heights)
            curve_points = pc[idx]
            current_pt = pc[pt_idx]
            constants = quadric_equation(curve_points[:, 0], curve_points[:, 1], curve_points[:, 2])
            # #TODO:把quadric_equation函数替换进去，不进行函数封装
            # X=curve_points[:,0]
            # Y=curve_points[:,1]
            # Z=curve_points[:,2]
            # X = X.reshape((len(X), 1))
            # Y = Y.reshape((len(Y), 1))
            # Z = Z.reshape((len(Z), 1))
            # num = len(X)
            #
            # # matrix = np.hstack((X**2, Y**2, Z**2, 2*X*Y, 2*X*Z, 2*Y*Z, 2*X, 2*Y, 2*Z)) #45%花费时间
            # matrix = np.hstack((X * X, Y * Y, Z * Z, 2 * X * Y, 2 * X * Z, 2 * Y * Z, 2 * X, 2 * Y, 2 * Z))
            # output = np.ones((num, 1))
            #
            # [a, b, c, d, e, f, g, h, i] = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(matrix), matrix)),
            #                                             np.transpose(matrix)), output)
            # a = -a[0]
            # b = -b[0]
            # c = -c[0]
            # d = -d[0]
            # e = -e[0]
            # f = -f[0]
            # g = -g[0]
            # h = -h[0]
            # i = -i[0]
            # j = 1
            #
            # constants = np.array([a, b, c, d, e, f, g, h, i, j])



            estimate_pt = getclosestpoint(current_pt, constants)
            k_values = compute_fundamentals(estimate_pt, constants)

            # point = estimate_pt
            # Fx = 2 * constants[0] * point[0] + constants[3] * point[1] + constants[5] * point[2] + constants[6]
            # Fy = 2 * constants[1] * point[1] + constants[3] * point[0] + constants[4] * point[2] + constants[7]
            # Fz = 2 * constants[2] * point[2] + constants[4] * point[1] + constants[5] * point[0] + constants[8]
            # Fxx = 2 * constants[0]
            # Fyy = 2 * constants[1]
            # Fzz = 2 * constants[2]
            # Fxy = constants[3]
            # Fyz = constants[4]
            # Fxz = constants[5]
            # grad_F = sqrt(Fx * Fx + Fy * Fy + Fz * Fz)
            #
            # # from https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.98.7059&rep=rep1&type=pdf
            # E = 1 + (Fx * Fx / Fz * Fz)
            # F = Fx * Fy / Fz * Fz
            # G = 1 + (Fy * Fy / Fz * Fz)
            # L = (1 / (Fz * Fz * grad_F)) * np.linalg.det(np.array(([Fxx, Fxz, Fx], [Fxz, Fzz, Fz], [Fx, Fz, 0])))
            # M = (1 / (Fz * Fz * grad_F)) * np.linalg.det(np.array(([Fxy, Fyz, Fy], [Fxz, Fzz, Fz], [Fx, Fz, 0])))
            # N = (1 / (Fz * Fz * grad_F)) * np.linalg.det(np.array(([Fyy, Fyz, Fy], [Fyz, Fzz, Fz], [Fy, Fz, 0])))
            #
            # A = np.array(([L, M], [M, N]))
            # B = np.array(([E, F], [F, G]))
            # B_inv = np.linalg.inv(B)
            #
            # k_values = np.linalg.eigvals(np.dot(B_inv, A))

            g_curvature = np.real(k_values[0] * k_values[1])  # K
            m_curvature = np.real(k_values[0] + k_values[1]) / 2
            cu_list = []
            cu_list.append(g_curvature)
            cu_list.append(m_curvature)
            angle_list.append(cu_list)

    return angle_list


def defect_Area(type):
    pd = select_type(pcd, type)
    pcdd = o3d.geometry.PointCloud()
    # 加载点坐标
    pcdd.points = o3d.utility.Vector3dVector(pd)
    print("->正在DBSCAN聚类...")
    # eps = 1.5  # 同一聚类中最大点间距
    eps = 1.0  # 同一聚类中最大点间距
    min_points = 3  # 有效聚类的最小点数
    labels = np.array(pcdd.cluster_dbscan(eps, min_points, print_progress=True))
    max_label = labels.max()  # 获取聚类标签的最大值 [-1,0,1,2,...,max_label]，label = -1 为噪声，因此总聚类个数为 max_label + 1
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # labels = -1 的簇为噪声，以黑色显示
    pcdd.colors = o3d.utility.Vector3dVector(colors[:, :3])




if __name__ == '__main__':

    # # pcd = o3d.io.read_point_cloud("piano_0015 - Cloud.pcd")     #更改为你想要读取点云的路径
    #
    # afile = 'fanzheng1'
    # txt_path = '../../txtcouldpoint/Final{}.txt'.format(afile)
    #
    #
    # # 通过numpy读取txt点云
    # pcd_1 = np.loadtxt(txt_path, delimiter=",")
    #
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pcd_1)
    # pcd = pcd.uniform_down_sample(50)
    pd, _ = display2()


    print(np.size(pd))
    # points[][]=pcd[:][:]
    # if pcd[:][3]==0:
    #     pd.append()
    # 筛选数据
    # print(pcd[:][0:2])
    pcd_point = pd[:][:, 0:3]
    d = pd[:][:, 0:3]
    print(pd)
    print(d)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_point)

    pc_as_np = np.asarray(pcd.points)
    st = time.time()
    # print(pcd)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    # results = compute_gaussian_curvature1(pc_as_np, pcd_tree,pd, k=20)
    results = compute_curvature1(pc_as_np,pcd_tree,pd,k=20)
    results_np1= np.array(results)
    et=time.time()
    print("time:")
    print(et-st)
    np.savetxt("Gauss_curvaturetest1.txt", results_np1)

