from numpy import *
from scipy.linalg import eig,svd,norm,schur
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import scipy.io as sio
from math import ceil
import sys

mat_contents = sio.loadmat('NLEVP/acoustic_wave_1d.mat')

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
R = 3.5 # Radius of large contour
R0 = R
R1 = R # Maximum radius of small contours without intersecting
Rfactor = 0.31 # multiplicative factor of the minimum distance from one eigenvalue to another 
mu = 0.j
mu1 = mu
K = 1

shift = True

def isincontour(z):
    tolcont = 1e-8
    return abs(z-mu)<R-tolcont



def T(z):
    
    if dim == 4:
        return (z**4)*a4+(z**3)*a3+(z**2)*a2+z*a1+a0
    if dim == 3:
        return (z**3)*a3+(z**2)*a2+z*a1+a0
    if dim == 2:
        return (z**2)*a2+z*a1+a0

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


""" routine starts here """

deletelist = []
for a in range(e.shape[0]):
    if not isincontour(e[a]):
        deletelist.append(a)
E = delete(e,deletelist)

mu = average(E)
mu1 = mu

RR = empty(len(E))
for i in range(len(E)):
    mindist = R1
    for j in range(len(E)):
        dist = abs(E[i]-E[j])*Rfactor
        if i!=j and dist < mindist:
            mindist = dist
    RR[i] = mindist
            

for contour in range(len(E)+1):
    if contour == 0:
        K = int(max(ceil(float(len(E))/m), 2))
        print "Main contour"
        print "N = "+str(N)+" ; K = "+str(K)+" ; R = "+str(R)+" ; mu = "+str(mu)
    else:
        K = 1 ; R = RR[contour-1] ; mu = E[contour-1]
        if contour == 1:
            print "Small contours"
            print "N = "+str(N)+" ; K = "+str(K)+" ; maximum R = "+str(R1)



    for l in range(lmin,m+1):
        
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
    tolres = 1e-2
    vects = dot(V01,svects)
    

    if not failed:
        for a in range(k):
            if not isincontour(lambs[a]):
                failed = True
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
                    failed = True
                    break
                
    if failed: # if it failed either of the two above checks then schur decompose
    
        try:
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
            Q = delete(Q,deletelist,1)
            
            svects = Q
            vects = dot(V01,svects) 
        except:
            lambs = array([])
    

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
            
            
    if contour == 0:
        l1 = lambs
        v1 = vects
        l2 = l1
        v2 = v1
        delno = 0
        if len(l1) == 0:
            print "No eigenvalues calculated for main contour, aborting"
            sys.exit()
        
    else:
        try:
            l2[contour-1-delno] = lambs[0]
            v2[:,contour-1-delno] = vects[:,0]
        except:
            delno += 1
            
if delno > 0:
    deletelist = range(len(l2)-delno,len(l2))
    l2 = delete(l2,deletelist)
    v2 = delete(v2,deletelist,1)
        
        
if len(l2) == 0:
    print "No eigenvalues calculated for small contours, aborting"
    sys.exit()
            
# error checks
# 1. error of the eigenvalues
error = 0
for i in range(l2.shape[0]):
    mindist = 2*R
    for j in range(l1.shape[0]):
        # check for nearest eigenvalue, then distance to it
        dist = abs(l2[i]-l1[j])
        if dist<mindist:
            mindist = dist
    error += mindist
error = error/l2.shape[0]
try:
    error = error[0]
except:
    pass
print "Total error in eigenvalues between both methods is "+str(error)

# 2. error of solution; is T(lambda)v = 0?
verror = 0
for a in range(l1.shape[0]):
    if sparse:
        test = norm(dot(T(l1[a]).todense(),v1[:,a]))
    else:
        try:
            test = norm(dot(T(l1[a]),v1[:,a]))
        except:
            test = norm(dot(T(l1[a]).todense(),v1[:,a]))
            sparse = True
    error += test
error = error/l1.shape[0]
try:
    error = error[0]
except:
    pass
print "Total error in solution of main contour is "+str(error)

verror = 0
for a in range(l2.shape[0]):
    if sparse:
        test = norm(dot(T(l2[a]).todense(),v2[:,a]))
    else:
        try:
            test = norm(dot(T(l2[a]),v2[:,a]))
        except:
            test = norm(dot(T(l2[a]).todense(),v2[:,a]))
            sparse = True
    error += test
error = error/l2.shape[0]
try:
    error = error[0]
except:
    pass
print "Total error in solution of small contours is "+str(error)


print "Number of eigenvalues found in main contour: "+str(len(l1))
print "Total number of eigenvalues found in small contours: "+str(len(l2))
print "Total eigenvalues within main contour: "+str(len(E))

def scatterplot():
    """ PLOT A GRAPH """
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    x1 = real(l1)
    y1 = imag(l1)

    x2 = real(l2)
    y2 = imag(l2)
    
    x3 = real(E)
    y3 = imag(E)
    
    ax1.scatter(x3, y3, marker='o', lw = 1.5, edgecolors = "r", facecolors = 'w', label = "$\lambda_{i,ML}}$")
    ax1.scatter(x1, y1, marker='x', c = "b", lw = 0.5, edgecolors = "b", label = "$\lambda_{i,IA2}^{(m)}$")
    ax1.scatter(x2, y2, marker='.', c = "g", lw = 0.5, edgecolors = "g", label = "$\lambda_{i,IA2}^{(e)}$")
    
    mur = real(mu1)
    mui = imag(mu1)
    
    plt.legend()
    
    ax1.add_artist(Circle((mur,mui),R0, fill=False, color='b'))
    plt.xlim(mur-R0,mur+R0)
    plt.ylim(mui-R0,mui+R0)
    
    for a in range(len(E)):
        mur = real(E[a])
        mui = imag(E[a])
        R = RR[a]
        ax1.add_artist(Circle((mur,mui), R, fill=False, color='g'))
        
    
    
    plt.show()
    
scatterplot()