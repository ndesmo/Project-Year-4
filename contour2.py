from numpy import *
from scipy.linalg import eig,svd,norm,schur
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import scipy.io as sio

mat_contents = sio.loadmat('NLEVP/bilby.mat')
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

lmin = m

N = 12
R = 100.
mu = 1.+1.j

def isincontour(z):
    tolcont = 1e-8
    return abs(z-mu)<R-tolcont

deletelist = []
for a in range(e.shape[0]):
    if not isincontour(e[a]):
        deletelist.append(a)
E = delete(e,deletelist)

K = max(len(E)/m + 1, 2)

print "N = "+str(N)+" ; K = "+str(K)+" ; R = "+str(R)+" ; mu = "+str(mu)

#lmin = m/K


#roots = array([1+1j,-1j,3j,-5+4j])

def T(z):
    
    if a3a4:
        return (z**4)*a4+(z**3)*a3+(z**2)*a2+z*a1+a0
    else:
        return (z**2)*a2+z*a1+a0
    """
    T = identity(m, dtype=complex)
    for root in roots:
        T = T*(z-root)
    return T
    """

def Phi(t):
    return mu + R*exp(1j*t)

def A(p):
    # compute A_p
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
    # compute the Hankel matrices
    B0 = zeros([K*m,K*l], dtype = complex)
    B1 = zeros([K*m,K*l], dtype = complex)
    Alist = []
    for i in range(2*K):
        Alist.append(A(i))
    for i in range(K):
        for j in range(K):
            B0[i*m:(i+1)*m,j*l:(j+1)*l]=Alist[i+j]
            B1[i*m:(i+1)*m,j*l:(j+1)*l]=Alist[i+j+1]
    return B0, B1

for l in range(lmin,m+1):
    
    Vhat = identity(m, dtype=complex)

    # contour integration
    
    B0, B1 = getBs(m,l)
    
    tolrank = 1e-10
    V, s, Wh = svd(B0) # do svd and calculate rank k
    k = sum(s>tolrank)
    if k!=K*l and k!=0:
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
tolres = 1e-2
vects = dot(V01,svects)

for a in range(k):
    if not isincontour(lambs[a]):
        failed = True
        print "Test: not in contour"
        break
    else:
        if sparse:
            test = norm(dot(T(lambs[a]).todense(),vects[:,a]))
        else:
            try:
                test = norm(dot(T(lambs[a]),vects[:,a]))
            except:
                test = norm(dot(T(lambs[a]).todense(),vects[:,a]))
                sparse = True
        if test>tolres:
            print "Test: invalid solution"
            failed = True
            break

if failed: # if it failed either of the two above checks then schur decompose

    U,Q,sdim = schur(D, output='complex', sort=isincontour) # schur decompose 
    
    lambs = zeros(k, dtype="complex")
    deletelist = []
    for a in range(k):
        lambs[a] = U[a,a]
        if not isincontour(lambs[a]):
            deletelist.append(a)
    lambs = delete(lambs,deletelist)
    U = delete(U,deletelist,0)
    Q = delete(Q,deletelist,1)
    
    svects = Q
    vects = dot(V01,svects)

deletelist = []

if lambs.shape[0] == 0:
    print "No eigenvalues computed within contour"
else:
    failed = False
    for a in range(lambs.shape[0]):
        if not isincontour(lambs[a]):            
            deletelist.append(a)
            failed = True
        else:
            if sparse:
                test = norm(dot(T(lambs[a]).todense(),vects[:,a]))
            else:
                try:
                    test = norm(dot(T(lambs[a]),vects[:,a]))
                except:
                    test = norm(dot(T(lambs[a]).todense(),vects[:,a]))
                    sparse = True
            if test>tolres:
                deletelist.append(a)
                failed = True
            
    if failed:
        lambs = delete(lambs,deletelist)
        vects = delete(vects,deletelist,1)
        print "Some values were incorrect, deleted"
    else:
        print "Values are correct"

if lambs.shape[0] == 0:
    print "No eigenvalues computed within contour"
else:
    
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
        error += mindist
    error = error/lambs.shape[0]
    try:
        error = error[0]
    except:
        pass
    print "Error in eigenvalues is "+str(error)
   
    # 2. error of solution; is T(lambda)v = 0?
    verror = 0
    for a in range(lambs.shape[0]):
        if sparse:
            test = norm(dot(T(lambs[a]).todense(),vects[:,a]))
        else:
            try:
                test = norm(dot(T(lambs[a]),vects[:,a]))
            except:
                test = norm(dot(T(lambs[a]).todense(),vects[:,a]))
                sparse = True
        error += test
    error = error/lambs.shape[0]
    try:
        error = error[0]
    except:
        pass
    print "Error in solution is "+str(error)


print "Number of eigenvalues found: "+str(len(lambs))
print "Total eigenvalues within contour: "+str(len(E))

#PLOT A GRAPH

fig = plt.figure()
ax1 = fig.add_subplot(111)

x1 = real(lambs)
y1 = imag(lambs)

x2 = real(lam)
y2 = imag(lam)

ax1.scatter(x1, y1, marker='o', c = "r")
ax1.scatter(x2, y2, marker='^', c = "b")

mur = real(mu)
mui = imag(mu)

ax1.add_artist(Circle((mur,mui),R, fill=False, color='g'))

plt.show()

