a = [3,4,1,7,2]

b=sorted(enumerate(a), key=lambda x:x[1])
print(b)