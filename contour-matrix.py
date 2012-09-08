from numpy import *

m = 1
l = 1
A = random.rand(m,m) + 1j*random.rand(m,m)-(1+1j)/2
#A[0,0]=0.3
Vhat = random.rand(m,l) + 1j*random.rand(m,l)
N = 10
R = 1.5
mu = 0.+0.j

def T(z):
    return z*identity(m, dtype=complex)-A

A0=zeros([m,l], dtype=complex)
dA0=zeros([m,l], dtype=complex)
for i in range(N):
    t = 2*i*pi/N
    B = (R/N)*Vhat*exp(1j*t)
    phi = mu + R*exp(1j*t)
    for k in range(l):
        dA0[:,k]=linalg.solve((T(phi)),B[:,k])

    A0 = A0 + dA0

print linalg.eig(A)[0]