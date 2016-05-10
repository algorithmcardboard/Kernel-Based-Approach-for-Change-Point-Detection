import generate_ts as ts
import numpy as np
from scipy.ndimage.interpolation import shift

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
    f= np.absolute(np.fft.fft(x));
    f=f[0:(n/2)]; 
    nf=f.shape[0];
    print('n is ', n, ' f shape is ', f.shape)

    f2=np.cumsum(f)/np.sum(f)
    print('f2 is ', np.sum(f2))
    f1 = shift(f2, 1, cval=0)
    c = np.zeros(nc+1)
    c[nc] = nf;
    for i in range(1, nc):
        c[i]=np.where((f1<=(i)/nc) & (f2>(i)/nc))[0][0]+1;

    lam = c/(2*c[nc])
    c = c - 1;

    cons=2*np.pi/frech;
    a=np.zeros((nc+1,n));

#ina=np.arange(0, int(n - 1))
    ina=np.arange(1, int(n))
    a[:,0]=lam*cons;
    a[:,1:n] = np.sin(np.matrix(lam).T * np.matrix(ina) *  cons)/(np.ones((nc+1, 1))*ina)
    da = np.diff(a, axis=0)
    A=np.zeros((n,n));
    for k in np.arange(nc):
        b=da[k,:]
        MU=np.zeros((n,n));
        for s in np.arange(n):
            xs=x[s];
            xjs=xs*xs*b[0];
            MU[s,s]=xjs;
            for j in np.arange(0,s-1):
                xjs=xjs+2*xs*x(s-j)*b(j+1);
                MU[s-j,s]=xjs;
        A=cumsum(MU,2);
        matD=matD - np.multiply(A,A);

    return matD


y = ts.get_random_ts('ts.csv')
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
