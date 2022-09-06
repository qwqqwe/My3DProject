import random
from line_profiler import LineProfiler
from functools import wraps


# 查询接口中每行代码执行的时间
def func_line_time(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        func_return = f(*args, **kwargs)
        lp = LineProfiler()
        lp_wrap = lp(f)
        lp_wrap(*args, **kwargs)
        lp.print_stats()
        return func_return

    return decorator


@func_line_time
# 定义一个测试函数
def random_sort2(n):
    l = [random.random() for i in range(n)]
    l.sort()
    return l


if __name__ == "__main__":
    random_sort2(2000000)