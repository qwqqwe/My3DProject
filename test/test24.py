import numpy
import perfplot


perfplot.show(
    setup=lambda n: numpy.random.randint(0, 1000, n),
    kernels=[
        lambda a: a[::-1],
        lambda a: numpy.ascontiguousarray(a[::-1]),
        lambda a: numpy.flipud(a)
        ],
    labels=['a[::-1]', 'ascontiguousarray(a[::-1])', 'flipud'],
    n_range=[2**k for k in range(25)],
    xlabel='len(a)',
    logx=True,
    logy=True,
    )