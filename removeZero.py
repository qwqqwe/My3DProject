import time

a=time.time()
with open('txtcouldpoint/Final/fan/fanfan2 (5).txt', 'r') as r:
    lines=r.readlines()
with open('txtcouldpoint/Finalfanfan5.txt', 'w') as w:
    for l in lines:
       if '0.00000' not in l:
          w.write(l)
b=time.time()
print(b-a)