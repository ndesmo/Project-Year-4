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

lmin = 1

A = random.rand(m,m) -1/2 + (random.rand(m,m) -1/2)*1j
#lam, vec = eig(A)
lam = e ; vec = X

N = 5
R = 10.
mu = 0.+0.j

def T(z):
    #return z*identity(m, dtype=complex)
    #return z*identity(m, dtype=complex)-A
    #return (z**4)*a4+(z**3)*a3+(z**2)*a2+z*a1+a0
    return (z**2)*a2+z*a1+a0

def isincontour(z):
    return abs(z-mu)<R

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
            try:
                dA[:,a]=linalg.solve((T(phi)),B[:,a])
            except:
                pass
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

A1+= mu*A0

V0 = V[:m,:k]       # trim matrices
W0h = Wh[:k,:l]
s = s[:k]

V0h = V0.transpose().conj()
W0 = W0h.transpose().conj()

Sinv = diag(1/s)

B = dot(dot(dot(V0h,A1),W0),Sinv) # calculate B

lambs,svects = eig(B) # calc eigenvalues and vectors of B, and A for comparison


print lambs

# are all eigenvalues in the contour?
failed = False
tolres = 1e-10
vects = zeros((m,k),dtype=complex)
for a in range(k):
    # eigenvalue within contour?
    if not isincontour(lambs[a]):
        failed = True
        print "outside range"
        continue
    else:
        # satisfies original equation?
        test = norm(dot(T(lambs[a]),dot(V0,svects[:,a])))
        if test>tolres:
            failed = True
            print "norm is",test


print failed

if failed: # if it failed either of the two above checks then schur decompose

    U,Q = schur(B, output='complex') # schur decompose 
    
    deletelist = []
    for a in range(k):
        if not isincontour(lambs[a]):
            deletelist.append(a)
    lambs = delete(lambs,deletelist)
    U = delete(U,deletelist,0)
    Q = delete(Q,deletelist,1)
    
    svects = Q

failed = False
for a in range(lambs.shape[0]):
    if not isincontour(lambs[a]):
        failed = True
        print "abs",lambs[a]
        #break
    else:
        test = norm(dot(T(lambs[a]),dot(V0,svects[:,a])))
        if test>tolres:
            failed = True
        print "tolres",test
            #break
        
print failed
print lambs
print lam

    