import generate_ts as ts
import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.linalg import toeplitz
from scipy.stats import t
from dcpc_functions import contfunc, dynaprog, pvalreg

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

y = ts.get_random_ts('ts.csv')
iop = 5
n = y.shape[0]
nmax = 1000
ld = 1
if(iop < 5):
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

