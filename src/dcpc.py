import generate_ts as ts
import numpy as np
from dcpc_functions import contfunc, dynaprog, pvalreg, moselci

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

Ke,M,PVAL,A = moselci(J,alpha,n)

print('changepoints are given by ', T[4, 0:3])
