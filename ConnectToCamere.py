from ctypes import *
import numpy as np

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
    listx = []
    for i in range(0, sizeoflen):
        if (x[i] == 0 or y[i] == 0 or z[i] == 0):
            None
        else:
            listx.append([x[i], y[i], z[i]])
    bbb = np.array(listx)
    targe1t.FreeMemory()
    return bbb
def Py_PrepareToCatch(targe1t):
    a=targe1t.PrepareToCatch()
    return a
def Py_Stop(targe1t):
    a=targe1t.Stop()
    return a

windll.LoadLibrary("C:/Users/Administrator/Documents/WeChat Files/wxid_bj5u8pz1th8e12/FileStorage/File/2022-08/v2.1.15.138(1)/G56N_SDK_DEMO_2.1.15.138_20220121_1748/CamWrapper/bins/X64/Debug/SgCamWrapper.dll")
targe1t=windll.LoadLibrary(r"C:\Users\Administrator\Documents\WeChat Files\wxid_bj5u8pz1th8e12\FileStorage\File\2022-08\v2.1.15.138(1)\G56N_SDK_DEMO_2.1.15.138_20220121_1748\CamWrapper\bins\X64\Debug\Dll6.dll")
# targe1t=windll.LoadLibrary(r"C:\Users\Administrator\source\repos\Dll6\x64\Debug\Dll6.dll")
#  完成Prepare之後再進行Catch，Catch可以進行多次，Prepare只用進行一次
#  出錯碼是
# -1相機連接失敗
# -2屬性設定失敗
# -3相機開啟失敗
# -4開始抓取失敗
# -5關閉相機失敗
print('return',Py_PrepareToCatch(targe1t))
bbb1=Py_Catch(targe1t)
bbb2=Py_Catch(targe1t)
bbb3=Py_Catch(targe1t)
print('return',Py_Stop(targe1t))
#完成Prepare之後再進行Catch，Catch可以進行多次，Prepare只用進行一次
# print('return',targe1t.Catch())
# # print('return',targe1t.Catch())
# # print('return',targe1t.Catch())
# #出錯碼是
# # -1相機連接失敗
# # -2屬性設定失敗
# # -3相機開啟失敗
# # -4開始抓取失敗
# # -5關閉相機失敗
# #這裡的函數是返回我們的數組，在Catch之後再抓，要不然會出錯
# targe1t.Reportx.restype = POINTER(c_float)
# targe1t.Reporty.restype = POINTER(c_float)
# targe1t.Reportz.restype = POINTER(c_float)
# targe1t.ReportSizeoflen.restype = c_int
# x=targe1t.Reportx()
# y=targe1t.Reporty()
# z=targe1t.Reportz()
# sizeoflen=targe1t.ReportSizeoflen()
# listx=[]
# numx = np.empty([sizeoflen,3], dtype = float)
# for i in range(0,sizeoflen):
#     if(x[i]==0 or y[i]==0 or z[i]==0):
#         None
#     else:listx.append([x[i],y[i],z[i]])
# bbb=np.array(listx)
# print(x[2], ',', y[2], ',', z[2])
# print(targe1t.FreeMemory())

















