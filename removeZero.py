import time

a=time.time()
with open('txtcouldpoint/Original/Third_874.txt', 'r') as r:
    lines=r.readlines()
with open('txtcouldpoint/Third_6.txt','w') as w:
    for l in lines:
       if '0.00000' not in l:
          w.write(l)
b=time.time()
print(b-a)