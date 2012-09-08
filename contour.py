from numpy import *

A = 0.5
N = 5
R = 1.
mu = 0.+0.j

def f(z):
    return log(z)

def contourint(f,N,R):
    itgl=0.+0.j
    for i in range(N):
        t = 2.*1j*pi/N
        itgl = itgl + R*exp(t*i)*f(R*exp(t))/N
    
    return itgl
        
print(contourint(f,N,R))