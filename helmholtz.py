#!/usr/bin/env python

# Help Python find the bempp module
import sys
sys.path.append("..")

from bempp.lib import *
from numpy import *
import scipy
from scipy.linalg import eig,svd,norm,schur,solve
from scipy.special import jv
from scipy.optimize import brentq


# Load mesh

grid = createGridFactory().importGmshGrid(
    "triangular", "helmholtz.msh")

# Create quadrature strategy

L = 2 # Order of quadrature
accuracyOptions = createAccuracyOptions()
# Increase by 2 the order of quadrature rule used to approximate
# integrals of regular functions on pairs on elements
accuracyOptions.doubleRegular.setRelativeQuadratureOrder(L)
# Increase by 2 the order of quadrature rule used to approximate
# integrals of regular functions on single elements
accuracyOptions.singleRegular.setRelativeQuadratureOrder(L)
quadStrategy = createNumericalQuadratureStrategy(
    "float64", "complex128", accuracyOptions)

# Create assembly context

assemblyOptions = createAssemblyOptions()
#assemblyOptions.switchToAcaMode(createAcaOptions())
context = createContext(quadStrategy, assemblyOptions)

# Initialize spaces

pwiseConstants = createPiecewiseConstantScalarSpace(context, grid)
pwiseLinears = createPiecewiseLinearContinuousScalarSpace(context, grid)

# Construct elementary operators

adjDblOp = createHelmholtz3dAdjointDoubleLayerBoundaryOperator(
    context, pwiseLinears, pwiseLinears, pwiseConstants, 2)
idOp = createIdentityOperator(
    context, pwiseLinears, pwiseLinears, pwiseLinears)


""" Eigenvalue solver is here """

# theoretical eigenvalues for cube
EN = 4
E = zeros(EN**3)
i = 0
for k1 in range(1,EN+1):
   for k2 in range(1,EN+1):
	for k3 in range(1,EN+1):
		E[i] = pi*sqrt(k1**2+k2**2+k3**2)
		i += 1

# get dimension of the matrix T
m = pwiseLinears.globalDofCount()

l = m

# initial parameters
N = 20
R = 10.0
mu = 11.+0.j

# paramter to shift the centre for greater numerical stability
shift = False

# parameter to keep only eigenvalues with small imaginary part
reals = False

def r(t):
    return R

def rdash(t):
    return 0.

def isincontour(z):
    tolcont = 1e-8
    return abs(z-mu)<r(angle(z-mu)) - tolcont

# initial parameters
Kmin = 1
Kmax = Kmin+2

def T(z):
    """ lambda matrix to solve """
    
    n = pwiseLinears.globalDofCount()
	
    adlpOp = createHelmholtz3dAdjointDoubleLayerBoundaryOperator(
        context, pwiseLinears, pwiseLinears, pwiseLinears, z)
    idOp = createIdentityOperator(
        context, pwiseLinears, pwiseLinears, pwiseLinears)
	
    TOp = 0.5*idOp - adlpOp
    
    TOpWeak = TOp.weakForm().asMatrix()
    return TOpWeak

def Phi(t):
    """ parametrization of the boundary """
    return mu + r(t)*exp(1j*t)

def PhiDash(t):
    """ derivative of the parametrization """
    return (rdash(t)+1j*r(t))*exp(1j*t)
    

def A(p):
    """ compute the pth moment """
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
        dA=solve(T(phi),B)
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

for K in range(Kmin,Kmax):
    # iteration begins
    if m == 1:
        Vhat = random.randn(m,l)
    else:
        Vhat = identity(m, dtype=complex)
        lmin = m

    # contour integration
    
    B0, B1 = getBs(m,l)
    
    # do svd and calculate rank k
    tolrank = 1e-10
    V, s, Wh = svd(B0) 
    k = sum(s>tolrank)
    if k!=K*l and k!=0:
        break
    
print "k = "+str(k)

# trim matrices
V0 = V[:K*m,:k] 
V01 = V[:m,:k]     
W0h = Wh[:k,:K*l]
s = s[:k]

V0h = V0.transpose().conj()
W0 = W0h.transpose().conj()

Sinv = diag(1/s)

# compute linearized matrix
D = dot(dot(dot(V0h,B1),W0),Sinv)

# compute eigenvalues of linearized matrix
try:
    lambs,svects = eig(D)
    if shift: lambs += mu
    failed = False
except:
    print "First try: No eigenvalues calculated"
    lambs = array([]) ; svects = array([])
    failed = True
# are all eigenvalues in the contour?
sparse = False
tolres = 1e-10
vects = dot(V01,svects)

deletelist = []

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
    """ find the partitions of the block upper triangular matrix T """
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
    """ Bartels stewart algorithm """
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
    """ compute the jordan form """
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
    U,Q,sdim = schur(D, output='complex', sort=isincontour)
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
    
# omit eigenvalues with large imaginary part / optional
if reals:
    deletelist = []
    for l in range(len(lambs)):
        if abs(imag(lambs[l]))>1e-1:
            deletelist.append(l)
        
    lambs = delete(lambs,deletelist)
    delete(vects,deletelist,1)


# check if eigenpairs are suitable
deletelist = []
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
   
    # 2. error of solution; is T(lambda)v = 0?
    error = 0.
    for a in range(lambs.shape[0]):
        error += norm(dot(T(lambs[a]),vects[:,a]))
    error = error/lambs.shape[0]
    print "Error in solution is "+str(error)


print "Number of eigenvalues found: "+str(len(lambs))
print lambs

# Compute resonant frequencies and damping constants
c = 340.29
f = zeros(len(lambs)) ; d = zeros(len(lambs))
rf = []
rl = []
rd = []
for l in range(len(lambs)):
    f[l] = c/(2*pi)*real(lambs[l])
    d[l] = -c*imag(lambs[l])
    if 	abs(imag(lambs[l])) < 1:
	rl.append(lambs[l])
	rf.append(f[l])
    	rd.append(d[l])
 
rf = array(rf)
rl = array(rl)
rd = array(rd)
print "Resonant wavenumbers"
print rl
print "Resonant frequencies"
print rf
    

def scatterplot():
    """ PLOT A GRAPH """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    
    x1 = real(lambs)
    y1 = imag(lambs)
    
    x2 = real(rl)
    y2 = imag(rl)
    
    xc = real(E)
    yc = imag(E)
    
    ax1.axhline(y=0, linestyle = '--', color = 'g')
    
    ax1.scatter(xc, yc, marker='s', c = "g", lw = 0.5, edgecolors = "g", label = "$\kappa_{cube}$")
    ax1.scatter(x1, y1, marker='.', c = "b", lw = 0.5, edgecolors = "b", label = "$\kappa_{IA2}$")
    ax1.scatter(x2, y2, marker='x', c = "r", lw = 0.5, edgecolors = "r", label = "$\kappa_{IA2}^{r}$")
    
    pts = arange(N)*2*pi/N
    Z = mu + r(pts)*exp(1j*pts)
    X1 = real(Z) ; Y1 = imag(Z)
    XY = zeros((N,2))
    XY[:,0] = X1 ; XY[:,1] = Y1
    ax1.add_artist(Polygon(XY, fill=False, color='g'))
    plt.xlim(min(X1), max(X1))
    plt.ylim(min(Y1), max(Y1))
    
    plt.xlabel("$Re\{\kappa\}$")
    plt.ylabel("$Im\{\kappa\}$")
    plt.title("Eigenvalue plot")
    
    plt.legend()
    
    ax2 = fig.add_subplot(212)
    
    x3 = rf
    y3 = rd
    
    ax2.scatter(x3, y3, marker='x', c = "r", lw = 0.5, edgecolors = "r", label = "$(f,\delta)$")
    
    ax2.xaxis.grid(color = 'gray', linestyle='-')
    ax2.axhline(y=0, linestyle = '--', color = 'g')
    plt.title("Resonant frequencies")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Damping constant")
    
    plt.legend()
    
    plt.show()
    
scatterplot()