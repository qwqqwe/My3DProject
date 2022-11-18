
from ctypes import *

class SGEXPORT_MODE(BigEndianStructure):
    _pack_ = 1
    _fields_ = [
        ('_ucCaptureMode',c_ubyte),
        ('_ucDataMode',c_ubyte),
        ('_ucTransMode',c_ubyte),
        ('_uiGrabNumber',c_uint32),
    ]
class SGEXPORT_MOD(BigEndianStructure):
    _pack_ = 1
    _fields_ = [
        ('_ucCaptureMode',c_ubyte),
        ('_ucTransMode',c_ubyte),
        ('_uiGrabNumber',c_uint),
    ]
class SGEXPORT_PROTOCL_ROI_ROW_OPTION(Structure):
    _pack_ = 1
    _fields_ = [
        ('_usRowStart',c_ushort),
        ('_usRowEnd',c_ushort),
    ]
class SGEXPORT_ROI(Structure):
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

        ('_bColumnSymmetry',c_ubyte),
        ('_ucColumnStep',c_ubyte),
        ('_usColumnLimit',c_ushort),

        ('_bRowSymmetry',c_ubyte),
        ('_ucRoiOptionCount',c_ubyte),

        ('_roiInfoReserve', c_ushort),

        ('_roiRowOption', SGEXPORT_PROTOCL_ROI_ROW_OPTION * 10),

        ('_ucZoomOptionCount',c_ubyte),
        ('_ucZoomOptions',c_ubyte *7),


    ]
class SGEXPORT_CFG(Structure):
    _pack_ = 1
    _fields_ = [
        ('_uiExpo',c_uint),
        ('_usThreshold',c_ushort),
        ('_fXScaling',c_float),
        ('_fYScaling',c_float),
        ('_fZScaling',c_float),
        ('_uiIp',c_uint),
        ('_ucDummyFilter',c_ubyte),
    ]
class SGEXPORT_PRODUCTINFO(Structure):
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
class SGEXPORT_ENCODER_PARAM(Structure):
    _pack_ = 1
    _fields_ = [
        ('_uiPulseNumber',c_uint),
        ('_isLowPrecisionEncoder',c_ubyte),
        ('_ucEncFilter',c_ubyte),
        ('_isDifferential',c_ubyte),
        ('_uiEncDiffJugdly', c_uint),
        ('_uiTriggerCount', c_uint),
    ]
class SGEXPORT_DEPTH_FILTER_WEIGHT(Structure):
    _pack_ = 1
    _fields_ = [
        ('_ucFilterStep',c_ubyte),
        ('_szStep3Weights',c_ubyte*3),
        ('_szStep5Weights',c_ubyte*5),
        ('_szStep7Weights',c_ubyte*7),
        ('_szStep9Weights',c_ubyte*9),
    ]
class SGEXPORT_CAMCONFIG(Structure):
    _pack_ = 1
    _fields_ = [
        ('_uiExpo',c_uint),
        ('_usFramePeriod',c_ushort),
        ('_usGain',c_ushort),
        ('_usThreshold',c_ushort),
        ('_ucAlgorithm',c_ubyte),
        ('_ucAlgorithmSize',c_ubyte),
        ('_isBlockAuto',c_ubyte),
        ('_ucBlockSel',c_ubyte),
        ('_isDummy',c_ubyte),
        ('_ucClockSel',c_ubyte),
        ('_uiIp',c_uint),
        ('_usPWM',c_ushort),
        ('_usColSelStart',c_ushort),
        ('_usColSelEnd',c_ushort),
        ('_usRowSel',c_ushort),
        ('_ucStepTH',c_ubyte),
        ('_ucFitZONE',c_ubyte),
        ('_ucFilterOption',c_ubyte),
        ('_ucDlyConf',c_ubyte),
        ('_fXScaling',c_float),
        ('_fYScaling',c_float),
        ('_fZScaling',c_float),
        ('_encParam',SGEXPORT_ENCODER_PARAM),
        ('_depthFilterWeight',SGEXPORT_DEPTH_FILTER_WEIGHT),
        ('_ucWindowEnable',c_ubyte),
        ('_ucWindowSize',c_ubyte),
        ('_ucMatchRate',c_ubyte),
        ('_ucLaserPower',c_ubyte),
        ('_fContourOffset',c_float),
        ('_fTanCoefficient',c_float),
        ('_ucRandomNoiseFilterThreshold',c_ubyte),
        ('_ucTBFilterEnable',c_ubyte),
        ('_ucDummyFilter',c_ubyte),
        ('_ucBadPointsThreshold',c_ubyte),
        ('_ucBottomNoiseFilter',c_ubyte),
        ('_ucLocationParam',c_ubyte),
        ('_ucFilterNoisyPointNum',c_ubyte),
        ('_ucEnableBatchPulse',c_ubyte),
        ('_usThresholdMaxValue',c_ushort),
        ('_usDebounceExtIO',c_ushort),
        ('_uiDebounceBatchIO',c_uint),
        ('_ucSpotMinInterval',c_ubyte),
    ]





target = windll.LoadLibrary("C:/Users/Administrator/Documents/WeChat Files/wxid_bj5u8pz1th8e12/FileStorage/File/2022-08/v2.1.15.138(1)/G56N_SDK_DEMO_2.1.15.138_20220121_1748/CamWrapper/bins/X64/Debug/SgCamWrapper.dll")
targe1t=windll.LoadLibrary(r"C:\Users\Administrator\source\repos\Dll6\x64\Debug\Dll6.dll")

# target.SgDetectNetCameras()
hCamera = target.SgCreateCamera()
target.SgConnect2CameraSyn(hCamera, b'169.254.215.237', b'169.254.215.0')

########################################################################
# camconfig=SGEXPORT_CAMCONFIG()
# target.SgGetConfigSyn(hCamera,camconfig)
#
# camconfig._usFramePeriod=1833
# camconfig._fYScaling=0.0384912
# # print(camconfig)
# target.SgSetCamConfigsSyn.argtypes=[c_uint,SGEXPORT_CAMCONFIG]
# target.SgSetCamConfigsSyn(hCamera,camconfig)
# camconfig=target.SgGetConfigSyn(hCamera,camconfig)
########################################################################

########################################################################
################################四位一体#################################
# roi=SGEXPORT_ROI()

roi=SGEXPORT_ROI()


targe1t.SGE_C_MODE()




target.SgGetRoiAndMediaSyn(hCamera,roi)
# mod=SGEXPORT_MOD()
targe1t.SGE_MOD.restype=SGEXPORT_MOD
mod=targe1t.SGE_MOD()
target.SgGetModSyn(hCamera,mod)
cfg=SGEXPORT_CFG()
# target.SgGetCfgSyn.argtypes=[c_uint,SGEXPORT_CFG]
target.SgGetCfgSyn(hCamera,cfg)
productinfo=SGEXPORT_PRODUCTINFO()
target.SgGetProductSyn(hCamera,productinfo)
################################四位一体#################################







########################################################################
# mod._ucCaptureMode=0
# mod._ucTransMode=1
# mod._uiGrabNumber=4000
# c_uint(4000)
# targe1t.SEG_CHANGE_MOD.restype=SGEXPORT_MOD
# mod=targe1t.SEG_CHANGE_MOD(c_ubyte(0),c_ubyte(1),c_uint(2000))
# mod._ucCaptureMode=c_ubyte(1)

targe1t.SGE_CHANGE_MODE.restype=SGEXPORT_MODE
mode=targe1t.SGE_CHANGE_MODE()
# mode=SGEXPORT_MODE()
target.SgGetModeSyn(hCamera,mode)
# mode=SGEXPORT_MODE()
print(target.SgSetCamModeSyn(hCamera,mode))
print(target.SgSetModSyn(hCamera,mod))#無效參數-2
target.SgSetModSyn(hCamera,mod)

test=SGEXPORT_MOD()
target.SgGetModSyn(hCamera,test)


target.SgStartCapture(hCamera)

target.SgSendGrabSignalToCamera(hCamera,bGrabEnd = False)
# target.SgSendGrabSignalToCamera(hCamera,bGrabEnd = False)
# target.SgSendGrabSignalToCamera(hCamera,bGrabEnd = False)
# target.SgSendGrabSignalToCamera(hCamera,bGrabEnd = False)
# target.SgSendGrabSignalToCamera(hCamera,bGrabEnd = False)
# target.SgSendGrabSignalToCamera(hCamera,bGrabEnd = False)

target.SgStopCaptureSyn(hCamera)

