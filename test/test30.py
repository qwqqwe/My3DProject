
from ctypes import *

class SGEXPORT_MODE(BigEndianStructure):
    _pack_ = 1
    _fields_ = [
        ('_ucCaptureMode',c_byte),
        ('_ucDataMode',c_byte),
        ('_ucTransMode',c_byte),
        ('_uiGrabNumber',c_uint),
    ]
class SGEXPORT_MOD(BigEndianStructure):
    _pack_ = 1
    _fields_ = [
        ('_ucCaptureMode',c_byte),
        ('_ucTransMode',c_byte),
        ('_uiGrabNumber',c_uint),
    ]
class SGEXPORT_PROTOCL_ROI_ROW_OPTION(BigEndianStructure):
    _pack_ = 1
    _fields_ = [
        ('_usRowStart',c_ushort),
        ('_usRowEnd',c_ushort),
    ]
class SGEXPORT_ROI(BigEndianStructure):
    _pack_ = 1
    _fields_ = [
        ('_usColumnStart',c_ushort),
        ('_usColumnEnd',c_ushort),
        ('_usRowStart',c_ushort),
        ('_usRowEnd',c_ushort),
        ('_usImgWidth',c_ushort),
        ('_usImgHeight',c_ushort),
        ('_usZoom',c_ushort),
        ('_reserve',c_ushort),
        ('_bColumnSymmetry',c_byte),
        ('_ucColumnStep',c_byte),
        ('_usColumnLimit',c_ushort),
        ('_bRowSymmetry',c_byte),
        ('_ucRoiOptionCount',c_byte),
        ('_roiInfoReserve', c_ushort),
        ('_roiRowOption', SGEXPORT_PROTOCL_ROI_ROW_OPTION * 10),
        ('_ucZoomOptionCount',c_byte),
        ('_ucZoomOptions',c_byte *7),


    ]
class SGEXPORT_CFG(BigEndianStructure):
    _pack_ = 1
    _fields_ = [
        ('_uiExpo',c_uint32),
        ('_usThreshold',c_ushort),
        ('_fXScaling',c_float),
        ('_fYScaling',c_float),
        ('_fZScaling',c_float),
        ('_uiIp',c_uint32),
        ('_ucDummyFilter',c_byte),
    ]
class SGEXPORT_PRODUCTINFO(BigEndianStructure):
    _pack_ = 1
    _fields_ = [
        ('_usProductParam',c_ushort),
        ('_usProductType',c_ushort),
        ('_usPcbVersion',c_ushort),
        ('_szFPGAVer',c_byte*32),
        ('_szEmbedVer',c_byte*32),
        ('_szSerialNumStr',c_byte*32),
        ('_szMACStr',c_byte*32),
        ('_szSensorType',c_byte*16),
    ]
class SGEXPORT_ENCODER_PARAM(BigEndianStructure):
    _pack_ = 1
    _fields_ = [
        ('_uiPulseNumber',c_uint32),# 编码器采样   z_num
        ('_isLowPrecisionEncoder',c_byte),# 是否低精度编码器
        ('_ucEncFilter',c_byte),# 编码器滤波参数
        ('_isDifferential',c_byte),# 是否差分
        ('_uiEncDiffJugdly', c_uint32),# 差分延时
        ('_uiTriggerCount', c_uint32),# 编码器计数
    ]
class SGEXPORT_DEPTH_FILTER_WEIGHT(BigEndianStructure):
    _pack_ = 1
    _fields_ = [
        ('_ucFilterStep',c_byte),# 是否低精度编码器
        ('_szStep3Weights',c_byte*3),# 编码器滤波参数
        ('_szStep5Weights',c_byte*5),# 是否差分
        ('_szStep7Weights',c_byte*7),# 是否差分
        ('_szStep9Weights',c_byte*9),# 是否差分
    ]
class SGEXPORT_CAMCONFIG(BigEndianStructure):
    _pack_ = 1
    _fields_ = [
        ('_uiExpo',c_uint32),# 曝光 shutter
        ('_usFramePeriod',c_ushort),#帧周期 frame_period
        ('_usGain',c_ushort),# 增益 gain
        ('_usThreshold',c_ushort),# 阈值 data_th
        ('_ucAlgorithm',c_byte),# 算法参数 arg_sel
        ('_ucAlgorithmSize',c_byte),# 算法 fac_sel
        ('_isBlockAuto',c_byte),# block_auto
        ('_ucBlockSel',c_byte),# block_sel
        ('_isDummy',c_byte),# dummy_en
        ('_ucClockSel',c_byte),# clk_sel
        ('_uiIp',c_uint32),#  ip， FPGA 中不保存了。由嵌入式软件中保存。
        ('_usPWM',c_ushort),#  laser_ctrl(laser_prep+laser_pwm)
        ('_usColSelStart',c_ushort),# col_sel_st
        ('_usColSelEnd',c_ushort),# col_sel_ed
        ('_usRowSel',c_ushort),# row_sel_lv
        ('_ucStepTH',c_byte),
        ('_ucFitZONE',c_byte),
        ('_ucFilterOption',c_byte),
        ('_ucDlyConf',c_byte),# dly_conf
        ('_fXScaling',c_float),
        ('_fYScaling',c_float),
        ('_fZScaling',c_float),
        ('_encParam',SGEXPORT_ENCODER_PARAM),# 编码器的一些参数设置
        ('_depthFilterWeight',SGEXPORT_DEPTH_FILTER_WEIGHT),# 补点算法设置
        ('_ucWindowEnable',c_byte),
        ('_ucWindowSize',c_byte),
        ('_ucMatchRate',c_byte),
        ('_ucLaserPower',c_byte),
        ('_fContourOffset',c_float),# 偏移
        ('_fTanCoefficient',c_float),# 倾斜纠正
        ('_ucRandomNoiseFilterThreshold',c_byte),# fil_para   随机噪点滤波阈值
        ('_ucTBFilterEnable',c_byte),# fil_en   顶底二级滤波使能
        ('_ucDummyFilter',c_byte),# data_th2   dummy平滑滤波
        ('_ucBadPointsThreshold',c_byte),# data_th3   坏点阈值
        ('_ucBottomNoiseFilter',c_byte),# 寄存器 0x2C，底噪滤波
        ('_ucLocationParam',c_byte),# 寄存器 0x2D，定位参数
        ('_ucFilterNoisyPointNum',c_byte),# 噪点滤波
        ('_ucEnableBatchPulse',c_byte),# 批处理脉冲使能【0：电平触发:1：脉冲触发】
        ('_usThresholdMaxValue',c_ushort),# 阈值最大值
        ('_usDebounceExtIO',c_ushort),# 电平触发信号滤波
        ('_uiDebounceBatchIO',c_uint32),# 批处理信号滤波
        ('_ucSpotMinInterval',c_byte),# 最小光斑间隔
    ]





target = windll.LoadLibrary("C:/Users/Administrator/Documents/WeChat Files/wxid_bj5u8pz1th8e12/FileStorage/File/2022-08/v2.1.15.138(1)/G56N_SDK_DEMO_2.1.15.138_20220121_1748/CamWrapper/bins/X64/Debug/SgCamWrapper.dll")


# target.SgDetectNetCameras()
hCamera = target.SgCreateCamera()
target.SgConnect2CameraSyn(hCamera, b'169.254.215.237', b'169.254.215.101')

########################################################################
camconfig=SGEXPORT_CAMCONFIG()
target.SgGetConfigSyn(hCamera,camconfig)
camconfig._usFramePeriod=1833
camconfig._fYScaling=0.0384912
target.SgSetCamConfigsSyn.argtypes=[c_int,SGEXPORT_CAMCONFIG]
target.SgSetCamConfigsSyn(hCamera,camconfig)
# camconfig=target.SgGetConfigSyn(hCamera,camconfig)
########################################################################

########################################################################
################################四位一体#################################
roi=SGEXPORT_ROI()
target.SgGetRoiAndMediaSyn(hCamera,roi)
mod=SGEXPORT_MOD()
target.SgGetModSyn(hCamera,mod)
cfg=SGEXPORT_CFG()
cfg._fYScaling=0.0384912
target.SgGetCfgSyn.argtypes=[c_int,SGEXPORT_CFG]
target.SgGetCfgSyn(hCamera,cfg)

productinfo=SGEXPORT_PRODUCTINFO()
target.SgGetProductSyn(hCamera,productinfo)
################################四位一体#################################







########################################################################
mod._ucCaptureMode=0
mod._ucTransMode=1
mod._uiGrabNumber=4000
target.SgSetModSyn(hCamera,mod)

target.SgStartCapture(hCamera)

target.SgSendGrabSignalToCamera(hCamera,bGrabEnd = False)

target.SgStopCaptureSyn(hCamera)

