import numpy as np
import json
import time


def mean_std(input):
  mm=np.mean(input)
  s = np.std(input)
  return mm,s

def statistics_poly():
    a = []
    f = open('test\\AllOutPutNom\\O9\\3.txt', 'r', encoding='utf-8')
    line = f.readline()
    for fields in line:
        fields = fields.strip('\n')
        fields = fields.strip('[')
        fields = fields.strip(']')
        data_split = fields.split(" ")
        temp = list(map(float,data_split))
        # fields = fields.split(",")
        a.append(temp)
    # while line:
    #     a.append(line.split('[]'))
    #     line =f.readline()

    f.close()
    print(a)




    # while line:
    #     a.append(line.split())  # 保存文件是以空格分离的
    #     line = f.readline()
    # f.close()
    # print(a)
    # b = np.asarray(a)
    # print(b)

def statistics_3():
    f = open('test\\AllOutPutNom\\O9\\3.txt', 'r', encoding='utf-8')
    lines = f.readlines()
    a = 0
    x3 = []
    x2 = []
    x1 = []
    x0 = []
    time_start = time.time()
    for line in lines:
        a += 1
        out = json.loads(line)
        x3.append(out[0])
        x2.append(out[1])
        x1.append(out[2])
        x0.append(out[3])
    mean3, std3 = mean_std(x3)
    mean2, std2 = mean_std(x2)
    mean1, std1 = mean_std(x1)
    mean0, std0 = mean_std(x0)

    print("mean3 =", mean3, "std3 =", std3)
    print("mean2 =", mean2, "std2 =", std2)
    print("mean1 =", mean1, "std1 =", std1)
    print("mean0 =", mean0, "std0 =", std0)
    time_end = time.time()
    print('Time cost = %fs' % (time_end - time_start))
    print(a)


def statistics_4():
    f = open('test\\AllOutPutNom\\O9\\4.txt', 'r', encoding='utf-8')
    lines = f.readlines()
    a = 0
    x4 = []
    x3 = []
    x2 = []
    x1 = []
    x0 = []
    time_start = time.time()
    for line in lines:
        a += 1
        out = json.loads(line)
        x4.append(out[0])
        x3.append(out[1])
        x2.append(out[2])
        x1.append(out[3])
        x0.append(out[4])
    mean4, std4 = mean_std(x4)
    mean3, std3 = mean_std(x3)
    mean2, std2 = mean_std(x2)
    mean1, std1 = mean_std(x1)
    mean0, std0 = mean_std(x0)
    print("mean4 =", mean4, "std4 =", std4)
    print("mean3 =", mean3, "std3 =", std3)
    print("mean2 =", mean2, "std2 =", std2)
    print("mean1 =", mean1, "std1 =", std1)
    print("mean0 =", mean0, "std0 =", std0)
    time_end = time.time()
    print('Time cost = %fs' % (time_end - time_start))
    print(a)
if __name__ == '__main__':
    # statistics_poly(
    statistics_3()
    statistics_4()