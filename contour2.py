from numpy import *
from scipy.linalg import eig,svd,norm,schur
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import scipy.io as sio

mat_contents = sio.loadmat('NLEVP/qep2.mat')
a0 = mat_contents['a0']
a1 = mat_contents['a1']
a2 = mat_contents['a2']
try:
    a3 = mat_contents['a3']
    a4 = mat_contents['a4']
    print "Quartic eigenvalue problem"
    a3a4 = True
except:
    print "Quadratic eigenvalue problem"
    a3a4 = False
e = mat_contents['e']
X = mat_contents['X']

m = a0.shape[0]
print "T is "+str(m)+" x "+str(m)

lmin = m/2

N = 10
K = 4
R = 8.
mu = 0.+0.j

print "N = "+str(N)+" ; K = "+str(K)+" ; R = "+str(R)+" ; mu = "+str(mu)



def T(z):
    if a3a4:
        return (z**4)*a4+(z**3)*a3+(z**2)*a2+z*a1+a0
    else:
        return (z**2)*a2+z*a1+a0

def isincontour(z):
    return abs(z-mu)<R

def Phi(t):
    return mu + R*exp(1j*t)

def A(p):
    sparse = False
    A0=zeros([m,l], dtype=complex)
    dA=zeros([m,l], dtype=complex)
    for i in range(N):
        t = 2*i*pi/N
        phi = Phi(t)
        B = (R/N)*Vhat*(phi**p)*exp(1j*t)
        for a in range(l):
            if sparse:
                dA[:,a]=spsolve(T(phi),B[:,a])
            else:
                try:
                    dA[:,a]=linalg.solve(T(phi),B[:,a])
                except:
                    dA[:,a]=spsolve(T(phi),B[:,a])
                    sparse = True
        A0 += dA
    return A0
        
def getBs(m,l):
    B = zeros([(K+1)*m,(K+1)*l], dtype = complex)
    for i in range(K+1):
        for j in range(K+1):
            B[i*m:(i+1)*m,j*l:(j+1)*l]=A(i+j)
    B0 = B[:(K)*m,:(K)*l]
    B1 = B[m:,l:]
    
    return B0, B1

for l in range(lmin,m+1):
    
    Vhat = identity(m, dtype=complex)

    # contour integration
    
    B0, B1 = getBs(m,l)
    
    tolrank = 1e-10
    V, s, Wh = svd(B0) # do svd and calculate rank k
    k = sum(s>tolrank)
    if k!=K*l:
        break
    
print "k = "+str(k)


V0 = V[:K*m,:k] 
V01 = V[:m,:k]     # trim matrices
W0h = Wh[:k,:K*l]
s = s[:k]

V0h = V0.transpose().conj()
W0 = W0h.transpose().conj()

Sinv = diag(1/s)

D = dot(dot(dot(V0h,B1),W0),Sinv) # calculate B

lambs,svects = eig(D) # calc eigenvalues and vectors of B, and A for comparison
lam = e ; vec = X
# are all eigenvalues in the contour?
failed = False
sparse = False
tolres = 1e-6
vects = zeros((k,k),dtype=complex)

for a in range(k):
    if not isincontour(lambs[a]):
        failed = True
        print "Test: not in contour"
        break
    else:
        if sparse:
            test = norm(dot(T(lambs[a]).todense(),dot(V01,svects[:,a])))
        else:
            try:
                test = norm(dot(T(lambs[a]),dot(V01,svects[:,a])))
            except:
                test = norm(dot(T(lambs[a]).todense(),dot(V01,svects[:,a])))
                sparse = True
        if test>tolres:
            print "Test: invalid solution"
            failed = True
            break

if failed: # if it failed either of the two above checks then schur decompose

    U,Q = schur(D, output='complex') # schur decompose 
    
    deletelist = []
    for a in range(k):
        if not isincontour(lambs[a]):
            deletelist.append(a)
    lambs = sqrt(delete(lambs,deletelist)) 
    """ NEED TO SORT OUT SQUARING ^ """
    U = delete(U,deletelist,0)
    Q = delete(Q,deletelist,1)
    
    svects = Q

failed = False
for a in range(lambs.shape[0]):
    if not isincontour(lambs[a]):
        failed = True
    else:
        if sparse:
            test = norm(dot(T(lambs[a]).todense(),dot(V01,svects[:,a])))
        else:
            try:
                test = norm(dot(T(lambs[a]),dot(V01,svects[:,a])))
            except:
                test = norm(dot(T(lambs[a]).todense(),dot(V01,svects[:,a])))
                sparse = True
        if test>tolres:
            failed = True
        
if failed:
    print "Some values are incorrect"
else:
    print "Values are correct"

# error checks
# 1. error of the eigenvalues
error = 0
for i in range(lambs.shape[0]):
    mindist = 2*R
    for j in range(lam.shape[0]):
        # check for nearest eigenvalue, then distance to it
        dist = abs(lambs[i]-lam[j])
        if dist<mindist:
            mindist = dist
    error += mindist**2
error = sqrt(error)/lambs.shape[0]
print "Error in eigenvalues is "+str(error[0])

# 2. error of solution; is T(lambda)v = 0?
verror = 0
for a in range(lambs.shape[0]):
    if sparse:
        test = norm(dot(T(lambs[a]).todense(),dot(V01,svects[:,a])))
    else:
        try:
            test = norm(dot(T(lambs[a]),dot(V01,svects[:,a])))
        except:
            test = norm(dot(T(lambs[a]).todense(),dot(V01,svects[:,a])))
            sparse = True
    error += test**2
error = sqrt(error)/lambs.shape[0]
print "Error in solution is "+str(error[0])
