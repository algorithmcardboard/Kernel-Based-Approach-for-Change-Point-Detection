import generate_ts as ts
import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.linalg import toeplitz
from scipy.stats import t

"""
    Constants
"""
SEGMENT_LENGTH = 15
segment_length = SEGMENT_LENGTH
MaX_SEGMENTS = 20
NMAX = 1000
alpha=5e-5

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
    matD.fill(np.inf)
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

    cons=2*np.pi/frech;
    a=np.zeros((nc+1,n));

    ina=np.arange(1, int(n))
    a[:,0]=lam*cons;
    a[:,1:n] = np.sin(np.matrix(lam).T * np.matrix(ina) *  cons)/(np.ones((nc+1, 1))*ina)
    da = np.diff(a, axis=0)
    A=np.zeros((n,n));
    for k in np.arange(int(nc)):
        b=da[k,:]
        MU=np.zeros((n,n));
        for s in np.arange(int(n)):
            xs=x[s];
            xjs=xs*xs*b[0];
            MU[s,s]=xjs;
            for j in np.arange(0,s):
                xjs=xjs+(2*xs*x[s-j-1]*b[j+1]);
                MU[s-j-1,s]=xjs;
        A=np.cumsum(MU,axis=1);
        matD= matD - np.multiply(A,A);

    matD = np.divide(matD, toeplitz(np.arange(1,n+1), np.arange(1,n+1)))

    if matD[0,n-1] < 0:
        matD=matD -2*matD[0,n-1] * (toeplitz(np.arange(1,n+1), np.arange(1,n+1))/n);


    I = matD.argmin(axis=0)
    Mmin = matD.min(axis=0)

    j = Mmin.argmin()
    MMin = Mmin.min()
    i=I[j];

    matD=matD - (MMin/(j-i+1))*toeplitz(np.arange(1,n+1), np.arange(1,n+1))

    return matD

def dynaprog(matD, Kmax):
    N,N = matD.shape
    I = np.empty((Kmax, N))
    I.fill(np.inf)

    t   = np.zeros((Kmax-1,N));
    I[0]=matD[0]
    for k in np.arange(1,Kmax-1):
        for L in np.arange(k, N):
            arr = I[k-1, np.arange(0,L)]+matD[np.arange(1,(L+1)),L]
            index = np.argmin(arr)
            val = arr[index]
            I[k, L] = val
            t[k-1, L] = index

    arr = I[Kmax-2, np.arange(0,N-1)] + matD[np.arange(1,N),N-1]
    index = np.argmin(arr)
    val = arr[index]

    I[Kmax-1,N-1] = val
    t[Kmax-2,N-1] = index
    J=I[:,N-1];

    t_est = np.diag(np.full(Kmax, N-1))

    for K in np.arange(1,Kmax):
        for k in np.arange(0, K)[K::-1]:
            t_est[K,k] = t[k,t_est[K,k+1]];

    return J, t_est;


def pvalreg(M,Y):
    n=Y.shape[0]-1;
    y= Y[1:n+1];
    yk=Y[0];
    # Construct regression matrix.
    V= M[1:n+1,:]
    vk=np.array(M[0]).flatten();
    #
    # Solve least squares problem, and save the Cholesky factor.
    [Q,R] = np.linalg.qr(V);
    a, residuals, rank, s = np.linalg.lstsq(R,np.array(np.matrix(y) * Q).reshape(3))
    r = y - np.array(V*a.reshape(3,1)).flatten()
    n,m = V.shape
    df = n - m
    normr = np.linalg.norm(r)
    yp=np.dot(vk,a);
    E = np.array(vk.T* R.T * (R*R.T).I).flatten()
    e = np.sqrt(1+np.multiply(E, E).sum())
    delta=yk-yp;
    pval = 1-1*t.cdf(delta*np.sqrt(df)/normr/e, df)
    f = np.vstack((vk, V))* np.matrix(a).T
    return pval,a,f


y = ts.get_random_ts('ts.csv')
n = y.shape[0]
nmax = 1000
ld = np.maximum(1,np.round(n/nmax))
print('ld after maximum is ', ld)
m= ld * np.fix(n/ld)
print("m is ",m)
xm=y[0:m]
Kmax = 20;

matD = contfunc(xm, SEGMENT_LENGTH, ld)

J, T = dynaprog(matD, Kmax)

T=T*ld;
T=T-np.diag(np.diag(T))+np.diag(n*np.ones(Kmax));

Kmax = J.shape[0];
pen= np.arange(1,Kmax+1)

PVAL = np.zeros(Kmax);
PVAL[0]=alpha;
A=np.zeros((Kmax-2,3));

for k in np.arange(1,Kmax-2):
    print(k);
    vx = np.arange(k, Kmax+1)
    V = np.hstack((np.ones((vx.shape[0], 1)), np.matrix(vx).reshape(vx.shape[0],1)))
    V = np.hstack( (V, (vx * np.log(n/vx)).reshape(vx.shape[0],1) ))

    pval,a,f = pvalreg(V,J[vx-1])
    PVAL[k]=pval
    A[k,:]=a

kv=[];
dv=[];
pv=[np.Inf];
dmax=1;
pmax=np.Inf;
k=0

while k<Kmax:
    pk=(J[k+1:Kmax]-J[k])/(pen[k]-pen[k+1:Kmax]);
    if(pk.shape[0] < 1):
        break;
    dm = pk.argmax();
    pm=pk.max()
    kv.append(k);
    dv.append(dm);
    pv.append(pm);
    if (dm>dmax and k>1):  
        dmax=dm; 
        pmax=pm
    print(k, dm, pm, dmax, pmax);
    k=k+dm+1;


lon = -1*np.diff(pv);

kv = np.array(kv)
pv = np.array(pv)

M = np.vstack((kv+1, pv[1:], pv[:-1], lon*-1)).T
Ko = kv.shape[0]

il= []
for i in np.arange(0,Ko-1):
    if lon[i]>np.max(lon[i+1:Ko]):
        il.append(i);


