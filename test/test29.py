import ctypes
target = ctypes.windll.LoadLibrary("C:/Users/Administrator/Documents/WeChat Files/wxid_bj5u8pz1th8e12/FileStorage/File/2022-08/v2.1.15.138(1)/G56N_SDK_DEMO_2.1.15.138_20220121_1748/CamWrapper/bins/X64/Debug/SgCamWrapper.dll")
# target1 = ctypes.windll.LoadLibrary("C:/Users/Administrator/Documents/WeChat Files/wxid_bj5u8pz1th8e12/FileStorage/File/2022-08/v2.1.15.138(1)/G56N_SDK_DEMO_2.1.15.138_20220121_1748/CamWrapper/bins/X64/Debug/tinyxml2.dll")

# target.SgDetectNetCameras()#检测相机是否存在
m_hCamera = target.SgCreateCamera()#创立一个新的相机实体
target.SgConnect2CameraSyn()
################################################################
# CBTYPE_COMMON = 0
# CBTYPE_CONNECT = 1
# CBTYPE_GET_MODE = 2
# CBTYPE_GET_CONFIG = 3
# CBTYPE_GET_BLOCKS = 4
# CBTYPE_GET_BLOCK = 5
# CBTYPE_GET_PRODINFO = 6
# CBTYPE_GET_ROI = 7
# CBTYPE_GET_REGSMAP = 8
# CBTYPE_REPORT_FRAMEPERIOD = 9
# CBTYPE_REPORT_TEMP = 10
# CBTYPE_CAP_IMG = 11
# CBTYPE_CAP_DEPTH = 12
# CBTYPE_CAP_PROGRESS = 13
# CBTYPE_REPORT_ERROR = 14
# CBTYPE_REPORT_TLV = 15
# CBTYPE_GET_MOD = 16
# CBTYPE_GET_CFG = 17
# CBTYPE_REPORT_COMMONDATA = 18
# CBTYPE_REPORT_MINPULSEINTERVAL = 20
################################################################

# target.SgRegCallback2(m_hCamera,)



# CBTYPE_REPORT_FRAMEPERIOD = 9 # 最小帧周期上报
# void CSingleCameraSynDialog::SG_FRAMEPERIOD_REPORT_CBFUNC(unsigned short usCommendFramePeriod, void *pOwner)
# {
# 	// usCommendFramePeriod 最小帧周期
# 	// 最大帧率  =  1000000 / usCommendFramePeriod;
# }
# target1.SG_FRAMEPERIOD_REPORT_CBFUNC
# target.SgRegCallback(m_hCamera, CBTYPE_REPORT_FRAMEPERIOD, target.SG_FRAMEPERIOD_REPORT_CBFUNC, target.SgRegCallback.this)
# target.SgRegCallback(m_hCamera, CBTYPE_CAP_DEPTH, SG_CAP_DEPTH_CBFUN, this)
# target.SgRegCallback(m_hCamera, CBTYPE_CAP_IMG, SG_CAP_PIC_CBFUN, this)
# CBTYPE_COMMON=0
