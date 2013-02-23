from numpy import *
from scipy.linalg import eig,svd,norm,schur
from scipy.sparse.linalg import spsolve
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import scipy.io as sio
from math import ceil
from scipy.special import cotdg

mat_contents = sio.loadmat('NLEVP/butterfly.mat')

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

l = m
Nmin = 5
Nmax = 14
Nstep = 1

Nrange = range(Nmin, Nmax+1, Nstep)
error1 = zeros(len(Nrange))
error2 = zeros(len(Nrange))

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

deletelist = []
for a in range(e.shape[0]):
    if not isincontour(e[a]):
        deletelist.append(a)
E = delete(e,deletelist)

Kmin = int(max(ceil(float(len(E))/m), 2))
Kmax = Kmin+1

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

for N in Nrange:
    for K in range(Kmin,Kmax):
        
        Vhat = identity(m, dtype=complex)
    
        # contour integration
        
        B0, B1 = getBs(m,l)
        
        tolrank = 1e-10
        V, s, Wh = svd(B0) # do svd and calculate rank k
        k = sum(s>tolrank)
        if k!=K*l and k!=0:
            break
        
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
        lambs = array([]) ; svects = array([])
        failed = True
    lam = e ; vec = X
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
            svects = inviteration(D, lambs, Q)
        else:
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
    
    if not lambs.shape[0] == 0:
        
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
        error1[(N-Nmin)/Nstep]=error
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
        error2[(N-Nmin)/Nstep]=error
    
    print "Number of eigenvalues found: "+str(len(lambs))
    print "Total eigenvalues within contour: "+str(len(E))

def convergenceplot():
    """ PLOT A GRAPH """
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    X = Nrange
    Y1 = error1
    Y2 = error2
    
    ax1.semilogy(X, Y1, label='$e_{eig,N}$')
    ax1.semilogy(X, Y2, label='$e_{sol,N}$')
    
    plt.xlabel('$N$')
    plt.ylabel('$e_N$')
    plt.title('Error graph')
    
    plt.legend()

    plt.show()
    
convergenceplot()