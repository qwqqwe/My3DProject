
def set(val,list=[]):
  list.append(val)
  return list

def foobar (a, b) :
    a+=1
    b.append(a)
    yield a +1
    a+=2
    b. append(a)
    yield a + 2
    # print(b[])


if __name__ == "__main__":
    # a = set(100)
    # b = set(200,[])
    # c = set(300)
    # print("a=%s" % a)
    # print("b=%s" % b)
    # print("c=%s" % c)
    a = 5
    b = []
    foobar(a,b)


    # for c in foobar(a,b):
    #     b.append(c)
    #
    # print(b)