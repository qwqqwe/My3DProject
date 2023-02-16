double_sort=[-1,0,3,5,9,12,15,16]
left=0
right= len(double_sort)-1
target=15
time=1
while (left<=right):
    middle=(left+right)//2
    if(double_sort[middle]>target):
        right=middle-1
    elif(double_sort[middle]<target):
        left=middle+1
    else:
        print(middle)
        break