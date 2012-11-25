from numpy import *
from scipy.linalg import eig,svd,norm,schur
import matplotlib.pyplot as plt
import scipy.io as sio

mat_contents = sio.loadmat('NLEVP/bicycle.mat')
a0 = mat_contents['a0']
a1 = mat_contents['a1']
a2 = mat_contents['a2']
#a3 = mat_contents['a3']
#a4 = mat_contents['a4']
e = mat_contents['e']
X = mat_contents['X']

m = a0.shape[0]

lmin = 1

A = random.rand(m,m) -1/2

N = 5
R = 10.0

mu = 0.+0.j

def T(z):
    #return z*identity(m, dtype=complex)
    #return z*identity(m, dtype=complex)-A
    #return (z**4)*a4+(z**3)*a3+(z**2)*a2+z*a1+a0
    return (z**2)*a2+z*a1+a0

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

V0 = V[:m,:k]       # trim matrices
W0h = Wh[:k,:l]
s = s[:k]

V0h = V0.transpose().conj()
W0 = W0h.transpose().conj()

Sinv = diag(1/s)

B = dot(dot(dot(V0h,A1),W0),Sinv) # calculate B

lambs,svects = eig(B) # calc eigenvalues and vectors of B, and A for comparison
lambs = lambs - mu
lam = e ; vec = X

print lambs

# are all eigenvalues in the contour?
failed = False
tolres = 1e-6
vects = zeros((m,k),dtype=complex)
deletelist = []
for a in range(k):
    if abs(lambs[a]-mu)>=R:
        failed = True
        deletelist.append(a)
        print "outside range"
        continue
    else:
        # satisfies original equation?
        vects[:,a] = dot(V0,svects[:,a])
        test = norm(dot(T(lambs[a]),vects[:,a]))
        if test>tolres:
            failed = True
            print "norm is",test
            deletelist.append(a)


print failed

print lam

if failed: # if it failed either of the two above checks then schur decompose

    U,Q = schur(B, output='complex') # schur decompose 
    
            
    lambs = delete(lambs,deletelist)
    print lambs
    Q = delete(Q,deletelist,1)
