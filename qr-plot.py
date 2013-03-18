import numpy as np
from numpy.linalg import inv, eig, norm
from scipy.linalg import qr
n = 10

# Initial matrices
A = 0.9*np.random.rand(n,n)
I = np.identity(n)
Inn = I[:,n-1]
InnT = Inn.transpose()

# set tolerance
toleig = 1e-1
# find eigenvalues to compare
L,V = eig(A)

tol = 1e-3

print "Computing eigenvalues of a "+str(n)+"x"+str(n)+" matrix A"

def T(z):
    return z*I - A

def Tdash(z):
    return I

def converged(nl,l):
    return abs(nl - l)<tol

def converged_eig(nl,e):
    return abs(nl - e)<toleig

N = 20
print "Maximum number of iterations: "+str(N)

pts = 50
    
X = np.linspace(-1.0, 1.0, pts)
Y = X

C = np.zeros((pts,pts))

total = pts*pts
current = 0.

""" iterate over grid """

a = 0
for x in X:
    b = 0
    for y in Y:
	nl = x + 1j*y
	l = nl + 1.

	i = 0

    """ perform QR algorithm """
	while i<N:
    	    if converged(nl,l): break
   	    try:
        	Q, R, P = qr(T(l), pivoting=True)
		P = np.diag(P)
        	l = nl
        	nl = l - 1/(np.dot(Inn,np.dot(Q.conj().transpose(),np.dot(Tdash(l),np.dot(P,np.dot(inv(R),InnT))))))
            except:
        	break
            i += 1
    
    # Colour regions according to closest eigenvalue
	k = 0
	for LL in L:
	    k += 1
    	    if abs(LL-l)<toleig:
                C[a,b] = k
                break

	b += 1
	current += 1
	if int(current/total*100) == current/total*100:
	    print str(int(current/total*100))+"% complete"
    a += 1

import matplotlib.pyplot as plt
# Plot a graph
plt.pcolor(X,Y,C.transpose())
plt.colorbar()
J = np.zeros((n+1), dtype = "complex")
J[1:] = L[:]
J[0] = 1e5
x = np.real(J) ; y = np.imag(J)
plt.scatter(x, y, marker = 'p', c = range(n+1), lw = 1, edgecolors = 'w')
plt.xlim([X.min(),X.max()])
plt.ylim([Y.min(),Y.max()])
plt.show()
