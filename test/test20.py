import time

import numpy as np

afile = 'zhengzheng1'
txt_path = '../txtcouldpoint/Final{}.txt'.format(afile)


save_path = "FinalOutPut/{}/".format(afile) + "Filter_{}.png"



start_time = time.time()
pcd_1 = np.loadtxt(txt_path, delimiter=",")
end_time = time.time()
print("Start",end_time-start_time)




start_time = time.time()
pcd_1 = np.genfromtxt(txt_path, delimiter=",")
end_time = time.time()
print("Start",end_time-start_time)