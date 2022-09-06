import numpy as np



def statistics_poly():
    a = []
    f = open('test\\AllOutPutNom\\O8\\3.txt', 'r', encoding='utf-8')
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

if __name__ == '__main__':
    statistics_poly()