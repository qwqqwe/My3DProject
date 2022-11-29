import numpy as np
from ctypes import *
class stPoint(BigEndianStructure):
    _pack_ = 1
    _fields_ = [
        ('x',c_float),
        ('y',c_float),
        ('z',c_float),
    ]


def fillprototype(f, restype, argtypes):
    f.restype = restype
    f.argtypes = argtypes


# main decleration
# fillprototype(mylib.dot_product, c_double, [POINTER(vector_double), POINTER(vector_double)])


target = windll.LoadLibrary("C:/Users/Administrator/Documents/WeChat Files/wxid_bj5u8pz1th8e12/FileStorage/File/2022-08/v2.1.15.138(1)/G56N_SDK_DEMO_2.1.15.138_20220121_1748/CamWrapper/bins/X64/Debug/SgCamWrapper.dll")
#targe1t=windll.LoadLibrary(r"C:\Users\Administrator\Documents\WeChat Files\wxid_bj5u8pz1th8e12\FileStorage\File\2022-08\v2.1.15.138(1)\G56N_SDK_DEMO_2.1.15.138_20220121_1748\CamWrapper\bins\X64\Debug\Dll6.dll")
targe1t=windll.LoadLibrary(r"C:\Users\Administrator\source\repos\Dll6\x64\Debug\Dll6.dll")
# targe2t=windll.LoadLibrary(r"C:\Users\Administrator\source\repos\ClassLibrary2\ClassLibrary2\bin\Debug\ClassLibrary2.dll")
# target.onDepth()
print('return',targe1t.PrepareToCatch())
print('return',targe1t.Catch())
targe1t.Reportx.restype = POINTER(c_float)
targe1t.Reporty.restype = POINTER(c_float)
targe1t.Reportz.restype = POINTER(c_float)
targe1t.ReportSizeoflen.restype = c_int
x=targe1t.Reportx()
y=targe1t.Reporty()
z=targe1t.Reportz()
sizeoflen=targe1t.ReportSizeoflen()
for i in range(0,sizeoflen):
    print(x[i])

# xx = np.array(np.fromiter(x, dtype=np.float64))
# yy = np.array(np.fromiter(y, dtype=np.float64))
# zz = np.array(np.fromiter(z, dtype=np.float64))
# print(xx)
# print('return',targe1t.SGE_CG_MODE())
# print('returne',targe2t.SENGO)














