# 引入numpy模块并创建两个向量x和y
import numpy as np
x=np.array((1,0,0))
y=np.array((2,1,1))

# 分别计算两个向量的模：
l_x=np.sqrt(x.dot(x))
l_y=np.sqrt(y.dot(y))
print('向量的模=',l_x,l_y)

# 计算两个向量的点积
dian=x.dot(y)
print('向量的点积=',dian)

# 计算夹角的cos值：
cos_=dian/(l_x*l_y)
print('夹角的cos值=',cos_)

# 求得夹角（弧度制）：
angle_hu=np.arccos(cos_)
print('夹角（弧度制）=',angle_hu)

# 转换为角度值：
angle_d=angle_hu*180/np.pi
print('夹角=%f°'%angle_d)
