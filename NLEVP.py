from numpy import *
from scipy.linalg import eig,svd,norm,schur
from scipy.sparse.linalg import spsolve
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import scipy.io as sio
from math import ceil
from scipy.special import cotdg

# Load the NLEVP problem

mat_contents = sio.loadmat('NLEVP/gen_tantipal2.mat')

# Ascertain the degree of the matrix polynomial

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

# initial parameters
l = m

N = 10
R = 0.4
mu = 0.+0.j

# Shift centre of contour for numerical stability
shift = True
# Set angle of rotation of the contour
ang = 0.
rad = ang*2*pi/360

def r(t):
    """ polar parametrization of contour """
    return (3+cos(5*(t+rad)))*R

def rdash(t):
    """ derivative of polar parametrization """
    return (-5*sin(5*(t+rad)))*R

def isincontour(z):
    """ routine to check whether a given value is inside the contour """
    tolcont = 1e-6
    return abs(z-mu)<r(angle(z-mu)) - tolcont

# Get the eigenvalues from MATLAB which are inside the contour
deletelist = []
for a in range(e.shape[0]):
    if not isincontour(e[a]):
        deletelist.append(a)
E = delete(e,deletelist)

# Set a range for K
Kmin = int(max(ceil(float(len(E))/m), 2))
Kmax = Kmin+1

print "N = "+str(N)+" ; Kmin = "+str(Kmin)+" ; Kmax: "+str(Kmax)+" ; R = "+str(R)+" ; mu = "+str(mu)

if shift: print "Shifted" 
else: print "Not shifted"

def T(z):
    """ matrix polynomial """
    if dim == 4:
        return (z**4)*a4+(z**3)*a3+(z**2)*a2+z*a1+a0
    if dim == 3:
        return (z**3)*a3+(z**2)*a2+z*a1+a0
    if dim == 2:
        return (z**2)*a2+z*a1+a0

def Phi(t):
    """ parametrization of z """
    return mu + r(t)*exp(1j*t)

def PhiDash(t):
    """ derivative of parametrization """
    return (rdash(t)+1j*r(t))*exp(1j*t)
    

def A(p):
    """ compute pth moment """
    sparse = False
    A0=zeros([m,l], dtype=complex)
    dA=zeros([m,l], dtype=complex)
    for i in range(N):
        t = 2*i*pi/N
        phi = Phi(t)
        phidash = PhiDash(t)
        if shift:
            B = (1/(N*1j))*Vhat*((phi-mu)**p)*phidash
        else:
            B = (1/(N*1j))*Vhat*(phi**p)*phidash
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
    """ compute the Hankel matrices """
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

print "Computing contour integrals"
for K in range(Kmin,Kmax):
    """ initiate iteration """
    Vhat = identity(m, dtype=complex)

    # contour integration
    
    B0, B1 = getBs(m,l)
    
    # do svd and calculate rank k
    tolrank = 1e-10
    V, s, Wh = svd(B0)
    k = sum(s>tolrank)
    if k!=K*l and k!=0:
        break
    
print "k = "+str(k)+" ; K = "+str(K)

# trim matrices
V0 = V[:K*m,:k] 
V01 = V[:m,:k]     
W0h = Wh[:k,:K*l]
s = s[:k]

V0h = V0.transpose().conj()
W0 = W0h.transpose().conj()

Sinv = diag(1/s)

# Calculate linearized matrix D
D = dot(dot(dot(V0h,B1),W0),Sinv)

# compute eigenvalues of D
try:
    lambs,svects = eig(D)
    if shift: lambs += mu
    failed = False
except:
    print "First try: No eigenvalues calculated"
    lambs = array([]) ; svects = array([])
    failed = True
lam = e ; vec = X
# are all eigenvalues in the contour?
sparse = False
tolres = 1e-10
vects = dot(V01,svects)

deletelist = []

# Check eigenpairs

if not failed:
    for a in range(k):
        if not isincontour(lambs[a]):
            deletelist.append(a)
    
    lambs = delete(lambs, deletelist)
    vects = delete(vects, deletelist, 1)
    
    for a in range(len(lambs)):
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

def inviteration(A, lambs, S, tol = 1e-8):
    """ inverse iteration algorithm """
    n = 20
    m = A.shape[0]
    k = S.shape[1]
    for j in range(k):
        l = lambs[j]
        s = S[:,j]
        F = (A-l*identity(m))
        for i in range(n):
            v = linalg.solve(F, s)
            s1 = v / norm(v, ord=inf)
            if norm(s1-s, ord=inf)<tol:
                break
            s = s1
        S[:,j] = s1
    return S

def partition(T, tol = 1e-8):
    """ ascertain the partition of the blocks of upper triangular matrix T """
    n = T.shape[0]
    l = 0.
    p = 1
    P = []
    for i in range(n):
        l1 = T[i,i]
        if abs(l1-l)<tol:
            p += 1
        else:
            P.append(p)
            p = 1
    return P

def bartels_stewart(F, G, C):
    """ Bartels-Stewart algorithm """
    p = C.shape[0] ; r = C.shape[1]
    Z = C
    for k in range(r):
        b = C[:,k]
        for s in range(p):
            for i in range(k-1):
                b[s] += G[i,k]*C[s,i]
        A = F - G[k,k]*identity(p)
        z = linalg.solve(F, b)
        Z[:,k] = z
    return Z

def golub_schur(Q, T):
    """ compute the Jordan form """
    P = partition(T)
    q = len(P)
    jsum = 0
    for j in range(1,q):
        isum = 0
        for i in range(j-1):
            jmin = jsum ; jmax = jsum + P[j]
            imin = isum ; imax = isum + P[i]
            Tii = T[imin:imin, imax:imax]
            Tjj = T[jmin:jmin, jmax:jmax]
            Tij = T[imin:imin, jmax:jmax]
            Z = bartels_stewart(Tii, Tjj, -Tij)
            ksum = 0
            for k in range(j,q):
                kmin = ksum ; kmax = ksum + P[k]
                Tik = T[imin:imin, kmax:kmax]
                Tjk = T[jmin:jmin, kmax:kmax]
                T[imin:imin, kmax:kmax] = Tik - dot(Z, Tjk)
                ksum += P[k]
            ksum = 0
            for k in range(j,q):
                kmin = ksum ; kmax = ksum + P[k]
                Qki = Q[kmin:kmin, imax:imax]
                Qkj = Q[kmin:kmin, jmax:jmax]
                Q[kmin:kmin, jmax:jmax] = dot(Z, Qki) - Qkj
                ksum += P[k]
            isum += P[i]
        jsum += P[j]
        
    return Q
        

if failed:
    # if it failed either of the two above checks then schur decompose
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
    U = delete(U,deletelist,1)
    #Q = delete(Q,deletelist,0)
    Q = delete(Q,deletelist,1)
    QH = zeros((k,k), dtype=complex)
    QH[:len(lambs),:len(lambs)]
    
    # compute eigenvectors
    
    if len(lambs) <= 0:
        print "Using inverse iteration"
        svects = inviteration(D, lambs, Q)
    else:
        print "Using Golub algorithm 7.6-3"
        svects = dot(QH, golub_schur(Q, U))
    
    vects = dot(V01,svects)

deletelist = []

# remove bad eigenpairs
if not lambs.shape[0] == 0:
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

def scatterplot():
    """ PLOT A GRAPH """
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    x1 = real(lambs)
    y1 = imag(lambs)

    x2 = real(E)
    y2 = imag(E)
    
    ax1.scatter(x2, y2, marker='s', s=120, lw = 0.5, edgecolors = "b", facecolors = 'w', label = "$\lambda_{predicted}$")
    ax1.scatter(x1, y1, marker='x', s=76,c = "b", lw = 1, edgecolors = "r", label = "$\lambda_{obtained}$")

    mur = real(mu)
    mui = imag(mu)
    
    pts = arange(N)*2*pi/N
    Z = mu + r(pts)*exp(1j*pts)
    X1 = real(Z) ; Y1 = imag(Z)
    XY = zeros((N,2))
    XY[:,0] = X1 ; XY[:,1] = Y1
    ax1.add_artist(Polygon(XY, fill=False, color='g'))
    plt.xlim(min(X1), max(X1))
    plt.ylim(min(Y1), max(Y1))
    
    plt.legend()
    
    plt.show()
    
scatterplot()