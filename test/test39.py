import time
from ctypes import *
import open3d as o3d
import numpy as np
import re

def Py_Catch(targe1t):
    targe1t.Catch()
    targe1t.Reportx.restype = POINTER(c_float)
    targe1t.Reporty.restype = POINTER(c_float)
    targe1t.Reportz.restype = POINTER(c_float)
    targe1t.ReportSizeoflen.restype = c_int
    x = targe1t.Reportx()
    y = targe1t.Reporty()
    z = targe1t.Reportz()
    sizeoflen = targe1t.ReportSizeoflen()
    xxx = np.empty([sizeoflen,3], dtype = float)
    xxx[:, 0]=x[:sizeoflen]
    xxx[:, 1]=y[:sizeoflen]
    xxx[:, 2]=z[:sizeoflen]
    targe1t.FreeMemory()
    return xxx
def Py_PrepareToCatch(targe1t):
    a=targe1t.PrepareToCatch()
    return a
def Py_Stop(targe1t):
    a=targe1t.Stop()
    return a

def Ip_Finder(str):
    result = re.findall(r'\D(?:\d{1,3}\.){3}(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\D', str)
    print(result)

    # 匹配开头可能出现ip
    ret_start = re.match(r'(\d{1,3}\.){3}(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\D', str)
    if ret_start:
        print("start:", ret_start.group())
        result.append(ret_start.group())

    # 匹配结尾
    ret_end = re.search(r'\D(\d{1,3}\.){3}(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)$', str)
    if ret_end:
        print("end: ", ret_end.group())
        result.append(ret_end.group())

    print('*' * 20)
    print("result: ", result)  # result:  ['g45.23.278.34h', 'f127.0.0.1j', 'j255.45.45.45b', '123.12.12.12']

    # 构造列表保存ip地址
    ip_list = []
    for r in result:
        # 正则提取ip
        ret = re.search(r'((25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)', r)
        if ret:
            # 匹配成功则将ip地址添加到列表中
            ip_list.append(ret.group())

    # 输入结果列表
    print(ip_list)
    return ip_list

def Py_Detect_IP(targe1t):
    targe1t.SGE_DETECENETCAM.restype = c_char_p
    std = targe1t.SGE_DETECENETCAM()
    str = bytes.decode(std)
    lista = Ip_Finder(str)
    return lista


windll.LoadLibrary("C:/Users/Administrator/Documents/WeChat Files/wxid_bj5u8pz1th8e12/FileStorage/File/2022-08/v2.1.15.138(1)/G56N_SDK_DEMO_2.1.15.138_20220121_1748/CamWrapper/bins/X64/Debug/SgCamWrapper.dll")
# targe1t=windll.LoadLibrary(r"C:\Users\Administrator\Documents\WeChat Files\wxid_bj5u8pz1th8e12\FileStorage\File\2022-08\v2.1.15.138(1)\G56N_SDK_DEMO_2.1.15.138_20220121_1748\CamWrapper\bins\X64\Debug\Dll6.dll")
targe1t=windll.LoadLibrary(r"C:\Users\Administrator\source\repos\Dll6\x64\Debug\Dll6.dll")
#  完成Prepare之後再進行Catch，Catch可以進行多次，Prepare只用進行一次
#  出錯碼是
# -1相機連接失敗
# -2屬性設定失敗
# -3相機設定失敗返回
# -4相機開啟失敗
# -5開始抓取失敗
# -6關閉相機失敗



lista=Py_Detect_IP(targe1t)#探測IP返回的是IP的list,
print('lista', lista)#######前面那個是本機，後面那個是相機
# lista[0]='1.1.1.1'
########連接時m_sHostIp(本機)是前面那個,m_CameraIP(相機)是後面那個
targe1t.PrepareToCatch.argtypes = [c_char_p,c_char_p]
hostip=lista[0].encode("UTF-8")
camip=lista[1].encode("UTF-8")
bhostip = create_string_buffer(hostip)
bcamip = create_string_buffer(camip)
return_prepare=targe1t.PrepareToCatch(bcamip,bhostip)#######在这里完成了相机的连接
print('return_prepare',return_prepare)


########开始设置参数，这个函数一定要有，但是参数不一定，因为我设定了默认值是3000
return_set=targe1t.SetInfo(300)
print('return_set',return_set)
########开始启动抓取
return_start=targe1t.StartCap()
print('return_start',return_start)
########开始抓取
bbb1=Py_Catch(targe1t)
# bbb2=Py_Catch(targe1t)

return_stop=targe1t.Stop()
print('return_stop',return_stop)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(bbb1)
pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pcd.points))
# 可视化
o3d.visualization.draw_geometries([pc_view], point_show_normal=True)