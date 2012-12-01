from numpy import *
from scipy.linalg import eig,svd,norm,schur
import matplotlib.pyplot as plt

Nmin = 1
Nmax = 10

mmin = 10
mmax = 16

eN = zeros((Nmax-Nmin,mmax-mmin))

for m in range(mmin,mmax):
    for N in range(Nmin,Nmax):
        #m = 6
        
        lmin = 1
        
        A = random.rand(m,m) -1/2
        
        #N = 3
        R = 10.0
        
        mu = 0.+0.j
        
        def T(z):
            #return z*identity(m, dtype=complex)
            return z*identity(m, dtype=complex)-A
        
        for l in range(lmin,m+1):
            
            Vhat = random.rand(m,l) + 1j*random.rand(m,l)
        
            # contour integration
            
            A0=zeros([m,l], dtype=complex)
            dA=zeros([m,l], dtype=complex)
            for i in range(N):
                t = 2*i*pi/N
                B = (R/N)*Vhat*exp(1j*t)
                phi = mu + R*exp(1j*t)
                for a in range(l):
                    dA[:,a]=linalg.solve((T(phi)),B[:,a])
                A0 += dA
            
            tolrank = 1e-10
            V, s, Wh = svd(A0) # do svd and calculate rank k
            k = sum(s>tolrank)
            if k!=l:
                break
            
        # compute A1
        
        A1=zeros([m,l], dtype=complex)
        dA=zeros([m,l], dtype=complex)
        for i in range(N):
            t = 2*i*pi/N
            B = (R**2/N)*Vhat*exp(2*1j*t)
            phi = mu + R*exp(1j*t)
            for a in range(l):
                dA[:,a]=linalg.solve((T(phi)),B[:,a])
            A1 += dA
        
        A1 += mu*A0
        
        V0 = V[:m,:k]       # trim matrices
        W0h = Wh[:k,:l]
        s = s[:k]
        
        V0h = V0.transpose().conj()
        W0 = W0h.transpose().conj()
        
        Sinv = diag(1/s)
        
        B = dot(dot(dot(V0h,A1),W0),Sinv) # calculate B
        
        lambs,svects = eig(B) # calc eigenvalues and vectors of B, and A for comparison
        lam,vec = eig(A)
        
        e = abs(norm(lambs)-norm(lam))
        eN[N-Nmin,m-mmin] = e

import matplotlib.pyplot as plt # Plotting a log graph of the results

x = arange(Nmin,Nmax)
y = eN
legendlist = []

for m in range(mmin,mmax):
    l = plt.semilogy(x,eN[:,m-mmin],basey=10)
    legendlist.append("m = "+str(m))

plt.xlabel("$N$")
plt.ylabel("$e_{N}$")
plt.title("Error plot")
plt.legend(legendlist)

plt.show()
