import ctypes
from enum import Enum
target = ctypes.windll.LoadLibrary("C:/Users/Administrator/Documents/WeChat Files/wxid_bj5u8pz1th8e12/FileStorage/File/2022-08/v2.1.15.138(1)/G56N_SDK_DEMO_2.1.15.138_20220121_1748/CamWrapper/bins/X64/Debug/SgCamWrapper.dll")
targe1t = ctypes.windll.LoadLibrary("C:/Users/Administrator/Documents/WeChat Files/wxid_bj5u8pz1th8e12/FileStorage/File/2022-08/v2.1.15.138(1)/G56N_SDK_DEMO_2.1.15.138_20220121_1748/CamWrapper/bins/X64/Debug/SgCamWrapper.dll")

# sgc=target.SgDetectNetCameras()
CBTYPE_COMMON = ctypes.c_int(0), # 通用响应回调
pCallback=ctypes.c_void_p
hCamera = target.SgCreateCamera()
# targe1t.CameraListener()
target.SgRegCallback2(hCamera,CBTYPE_COMMON,pCallback)