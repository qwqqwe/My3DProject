import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import scipy.signal as signal
x=np.array([
    0, 6, 25, 20, 15, 8, 15, 6, 0, 6, 0, -5, -15, -3, 4, 10, 8, 13, 8, 10, 3,
    1, 20, 7, 3, 0 ])
plt.figure(figsize=(16,4))
plt.plot(np.arange(len(x)),x)
print (x[signal.argrelextrema(x, np.greater)])
print (signal.argrelextrema(x, np.greater))

plt.plot(signal.argrelextrema(x,np.greater)[0],x[signal.argrelextrema(x, np.greater)],'o')
plt.plot(signal.argrelextrema(-x,np.greater)[0],x[signal.argrelextrema(-x, np.greater)],'+')
# plt.plot(peakutils.index(-x),x[peakutils.index(-x)],'*')
plt.show()