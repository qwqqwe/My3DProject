import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import csv
import openpyxl

def txt_to_csv():
    a = []
    f = open('test\\AllOutPutNom\\O8\\2.txt','r',encoding='utf-8')
    line = f.readline()
    while line:
        a.append(line.split())#保存文件是以空格分离的
        line = f.readline()
    f.close()
    fp = open('follow_name_1.csv','w',encoding='utf_8_sig',newline="")
    csvwriter=csv.writer(fp)
    csvwriter.writerows(a)
    print("finish")


def parallel():
    data = pd.read_csv('follow_name_1.csv')

    plt.figure('多维度-parallel_coordinates')
    plt.title('parallel_coordinates')
    parallel_coordinates(data, 'NULL', color=['blue', 'green', 'red', 'yellow'])
    plt.show()

def txt_to_xlsx():
    input='test\\AllOutPutNom\\O8\\2.txt'
    output = 'test\\AllOutPutNom\\O8\\2.xlsx'
    wb = openpyxl.Workbook()
    ws = wb.worksheets[0]
    with open(input,'rt',encoding="utf-8") as data:
        reader = csv.reader(data,delimiter=' ')
        for row in reader:
            ws.append(row)
    wb.save(output)

if __name__ == '__main__':
    # txt_to_csv()
    # parallel()
    txt_to_xlsx()