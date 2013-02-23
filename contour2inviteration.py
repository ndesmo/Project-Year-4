from numpy import *
from scipy.linalg import eig,svd,norm,schur
from scipy.sparse.linalg import spsolve
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import scipy.io as sio
from math import ceil

mat_contents = sio.loadmat('NLEVP/qep1.mat')

dim = 2
a0 = mat_contents['a0']
a1 = mat_contents['a1']
a2 = mat_contents['a2']
try:
    a3 = mat_contents['a3']
    try:
        a4 = mat_contents['a4']
        dim = 4
        print "Quartic eigenvalue problem"
    except:
        dim = 3
        print "Cubic eigenvalue problem"
except:
    print "Quadratic eigenvalue problem"
    
e = mat_contents['e']
X = mat_contents['X']

m = a0.shape[0]
print "T is "+str(m)+" x "+str(m)

lmin = m

N = 5
R = 1.
mu = 1.+0.j

shift = True

def isincontour(z):
    tolcont = 1e-8
    return abs(z-mu)<R-tolcont

deletelist = []
for a in range(e.shape[0]):
    if not isincontour(e[a]):
        deletelist.append(a)
E = delete(e,deletelist)

K = int(max(ceil(float(len(E))/m), 2))

print "N = "+str(N)+" ; K = "+str(K)+" ; R = "+str(R)+" ; mu = "+str(mu)

if shift: print "Shifted" 
else: print "Not shifted"

#lmin = m/K


#roots = array([1+1j,-1j,3j,-5+4j])

def T(z):
    
    if dim == 4:
        return (z**4)*a4+(z**3)*a3+(z**2)*a2+z*a1+a0
    if dim == 3:
        return (z**3)*a3+(z**2)*a2+z*a1+a0
    if dim == 2:
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
        if shift:
            B = (R/N)*Vhat*((phi-mu)**p)*exp(1j*t)
        else:
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

try:
    lambs,svects = eig(D) # calc eigenvalues and vectors of B, and A for comparison
    if shift: lambs += mu
    failed = False
except:
    print "First try: No eigenvalues calculated"
    lambs = array([]) ; svects = array([])
    failed = True
lam = e ; vec = X
# are all eigenvalues in the contour?
sparse = False
tolres = 1e-2
vects = dot(V01,svects)

if not failed:
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

def inverse_iteration(A, l, tolerance):
    n = A.shape[0]
    x = ones(n)
    Ashift = A - l*identity(n)
    error = tolerance + 1
    while error > tolerance:
        x = sp.linalg.solve(Ashift, x)
        x = x / sp.linalg.norm(x)
        error = sp.linalg.norm(l*x - dot(A, x))
    return x

if failed: # if it failed either of the two above checks then schur decompose

    U,Q,sdim = schur(D, output='complex', sort=isincontour) # schur decompose
    lambs = zeros(k, dtype="complex")
    deletelist = []
    for a in range(k):
        lambs[a] = U[a,a]
        if shift: lambs[a] += mu
        if not isincontour(lambs[a]):
            deletelist.append(a)
    lambs = delete(lambs,deletelist)
    U = delete(U,deletelist,0)
    R = delete(Q,deletelist,1)
    
    P = zeros((k,len(lambs)), dtype = complex)
    for i in range(len(lambs)):
        P[:,i] = inverse_iteration(Q, lambs[i], 1e-)
    print P
    
    svects = P
    vects = dot(V01,svects)
    

deletelist = []

if not lambs.shape[0] == 0:
    failed = False
    for a in range(lambs.shape[0]):
        print lambs
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

def scatterplot():
    """ PLOT A GRAPH """
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    x1 = real(lambs)
    y1 = imag(lambs)

    x2 = real(E)
    y2 = imag(E)
    
    ax1.scatter(x2, y2, marker='o', lw = 1.5, edgecolors = "r", facecolors = 'w', label = "$\lambda_{i,ML}}$")
    ax1.scatter(x1, y1, marker='.', c = "b", lw = 0.5, edgecolors = "b", label = "$\lambda_{i,IA2}$")

    mur = real(mu)
    mui = imag(mu)
    
    ax1.add_artist(Circle((mur,mui),R, fill=False, color='g'))
    
    plt.legend()
    
    plt.show()
    
scatterplot()