import time
from ctypes import *
import numpy as np
from line_profiler import LineProfiler
from functools import wraps
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
@func_line_time
# 定义一个测试函数
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
windll.LoadLibrary("C:/Users/Administrator/Documents/WeChat Files/wxid_bj5u8pz1th8e12/FileStorage/File/2022-08/v2.1.15.138(1)/G56N_SDK_DEMO_2.1.15.138_20220121_1748/CamWrapper/bins/X64/Debug/SgCamWrapper.dll")
# targe1t=windll.LoadLibrary(r"C:\Users\Administrator\Documents\WeChat Files\wxid_bj5u8pz1th8e12\FileStorage\File\2022-08\v2.1.15.138(1)\G56N_SDK_DEMO_2.1.15.138_20220121_1748\CamWrapper\bins\X64\Debug\Dll6.dll")
targe1t=windll.LoadLibrary(r"C:\Users\Administrator\source\repos\Dll6\x64\Debug\Dll6.dll")
#  完成Prepare之後再進行Catch，Catch可以進行多次，Prepare只用進行一次
#  出錯碼是
# -1相機連接失敗
# -2屬性設定失敗
# -3相機開啟失敗
# -4開始抓取失敗
# -5關閉相機失敗
print('return',Py_PrepareToCatch(targe1t))
a=time.time()
bbb1=Py_Catch(targe1t)
bbb2=Py_Catch(targe1t)
# bbb3=Py_Catch(targe1t)
b=time.time()
# print(b-a)
print('return',Py_Stop(targe1t))
















