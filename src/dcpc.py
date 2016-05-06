import generate_ts as ts
import numpy as np


"""
    Constants
"""
SEGMENT_LENGTH = 15
MaX_SEGMENTS = 20
NMAX = 1000

"""
    END constants
"""

def contfunc(y, segment_length, ld):
    ld = 1
    nd=y.shape[0]
    n=np.round(nd/ld)
    print("n is ", n, " ld is ", ld)
    lmind=np.ceil(segment_length/ld)
    matD = np.empty((n, n))
    print('matD shape is ', matD.shape)
    matD.fill(np.inf)
    print('matD shape is ', matD.shape)
    matD=np.tril(matD,lmind-2)

    x = y - np.mean(y)
    print(x.shape)

    frech = 1;
    nc=5
    f= np.absolute(np.fft.fft(x))
    f=f[0:(n/2)]; nf=f.shape[0];
    print('n is ', n, ' f shape is ', f.shape)

    f2=np.cumsum(f)/np.sum(f)
    print('f2 is ', np.sum(f2))

    return matD


y = ts.get_random_ts()
n = y.shape[0]
nmax = 1000
ld = np.maximum(1,np.round(n/nmax))
print('ld after maximum is ', ld)
m= ld * np.fix(n/ld)
print("m is ",m)
xm=y[0:m]

matD = contfunc(xm, SEGMENT_LENGTH, ld)

#print(y.shape)
#print(xm.shape)
