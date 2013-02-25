#!/usr/bin/env python

# Help Python find the bempp module
import sys
sys.path.append("/home/nick/bempp/python")

from bempp.lib import *
from numpy import *
import scipy
from scipy.linalg import eig,svd,norm,schur

# Load mesh

grid = createGridFactory().importGmshGrid(
    "triangular", "./sphere-h-0.2.msh")

# Create quadrature strategy

accuracyOptions = createAccuracyOptions()
# Increase by 2 the order of quadrature rule used to approximate
# integrals of regular functions on pairs on elements
accuracyOptions.doubleRegular.setRelativeQuadratureOrder(2)
# Increase by 2 the order of quadrature rule used to approximate
# integrals of regular functions on single elements
accuracyOptions.singleRegular.setRelativeQuadratureOrder(2)
quadStrategy = createNumericalQuadratureStrategy(
    "float64", "complex128", accuracyOptions)

# Create assembly context

assemblyOptions = createAssemblyOptions()
#assemblyOptions.switchToAcaMode(createAcaOptions())
context = createContext(quadStrategy, assemblyOptions)

# Initialize spaces

pwiseConstants = createPiecewiseConstantScalarSpace(context, grid)
pwiseLinears = createPiecewiseLinearContinuousScalarSpace(context, grid)



""" Eigenvalue solver is here """

m = pwiseConstants.globalDofCount()

l = m

N = 12
R = 0.4
mu = 0.+0.j


shift = True
ang = 180.
rad = ang*2*pi/360

def r(t):
    return (3+cos(2*(t+rad)))*R
    #return (exp(2*cos(t+rad))+1)*R

def rdash(t):
    return (-2*sin(2*(t+rad)))*R
    #return (-sin(2*(t+rad))*exp(2*cos(t+rad)))*R

def isincontour(z):
    tolcont = 1e-8
    #return abs(z-mu)<R-tolcont
    return abs(z-mu)<r(angle(z-mu)) - tolcont

Kmin = 2
Kmax = Kmin+1

print "N = "+str(N)+" ; Kmin = "+str(Kmin)+" ; Kmax: "+str(Kmax)+" ; R = "+str(R)+" ; mu = "+str(mu)

if shift: print "Shifted" 
else: print "Not shifted"

#lmin = m/K


#roots = array([1+1j,-1j,3j,-5+4j])

def T(z):
    slpOp = createHelmholtz3dSingleLayerBoundaryOperator(
        context, pwiseConstants, pwiseLinears, pwiseConstants,z)

    slpWeak = slpOp.weakForm().asMatrix()
    return slpWeak

def Phi(t):
    #return mu + R*exp(1j*t)
    return mu + r(t)*exp(1j*t)

def PhiDash(t):
    return (rdash(t)+1j*r(t))*exp(1j*t)
    

def A(p):
    # compute A_p
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

for K in range(Kmin,Kmax):
    
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
    p = C.shape[0] ; r = C.shape[1]
    Z = C
    for k in range(r):
        b = c[:,k]
        for s in range(p):
            for i in range(k-1):
                b[s] += G[i,k]*C[s,i]
        #b = c[:,k] + dot(C, G[:,k])
        A = F - g[k,k]*identity(p)
        z = linalg.solve(F, b)
        Z[:,k] = z
    return Z

def golub_schur(Q, T):
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
    U = delete(U,deletelist,1)
    #Q = delete(Q,deletelist,0)
    Q = delete(Q,deletelist,1)
    QH = zeros((k,k), dtype=complex)
    QH[:len(lambs),:len(lambs)]
    
    
    if len(lambs) <= k/4:
        print "Using inverse iteration"
        svects = inviteration(D, lambs, Q)
    else:
        print "Using Golub algorithm 7.6-3"
        svects = dot(QH, golub_schur(Q, U))
    
    vects = dot(V01,svects)

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