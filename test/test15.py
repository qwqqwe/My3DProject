import numpy as np
# txt_path = 'FinalOutPut/distroy1/50none.txt'
# with open('FinalOutPut/distroy1/50none.txt', 'r') as r:
#     lines = r.
# print(lines)
lines=np.genfromtxt('FinalOutPut/distroy1/50none.txt')
print(lines)
if len(lines) != 0:
    starta = lines[0]
    for line in range(len(lines) - 1):
        if (lines[line] + 1 != lines[line + 1]):
            enda = lines[line]
            print('start:', starta, 'end:', enda)
            starta = lines[line + 1]