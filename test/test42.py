double_sort=[-1,0,3,5,0,9,0,12,0,15,16,-1]
left=0
right= len(double_sort)-1
for i in range(right):
    if double_sort[i]!=0:
        double_sort[left]=double_sort[i]
        left+=1
print(double_sort)