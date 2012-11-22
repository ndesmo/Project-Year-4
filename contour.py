from numpy import *
from scipy.linalg import eig,svd,norm,schur
import matplotlib.pyplot as plt
import scipy.io as sio

mat_contents = sio.loadmat('NLEVP/qep1.mat')
a0 = mat_contents['a0']
a1 = mat_contents['a1']
a2 = mat_contents['a2']
#a3 = mat_contents['a3']
#a4 = mat_contents['a4']
e = mat_contents['e']
X = mat_contents['X']

m = a0.shape[0]

lmin = m

A = random.rand(m,m) -1/2

N = 40
R = 5.0

mu = 0.+0.j

def T(z):
    #return z*identity(m, dtype=complex)
    #return z*identity(m, dtype=complex)-A
    #return (z**4)*a4+(z**3)*a3+(z**2)*a2+z*a1+a0
    return (z**2)*a0+z*a1+a2

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
    
print k
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
lam = e ; vec = X
print lambs
print e

# are all eigenvalues in the contour?
failed = False
tolres = 1e-6
vects = zeros((k,k),dtype=complex)
for a in range(k):
    if abs(lambs[a]-mu)>=R:
        failed = True
        #print "abs",lambs[a]
        break
    else:
        vects[:,a] = dot(V0,svects[:,a])
        test = norm(dot(T(lambs[a]),vects[:,a]))
        if test>tolres:
            failed = True
            #print "tolres",test
            break

"""        
print lambs, norm(vects)
print lam,norm(vec)
print sqrt(m)
"""
print failed

if failed: # if it failed either of the two above checks then schur decompose

    U,Q = schur(B, output='complex') # schur decompose 
    
    deletelist = []
    for a in range(k):
        if abs(lambs[a]-mu)>=R:
            deletelist.append(a)
    lambs = delete(a,deletelist)
    Q = delete(Q,deletelist,1)
    
    #print Q
            

