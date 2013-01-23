from numpy import *
from scipy.linalg import eig,svd,norm,schur
    
def eigSolve(VV, K=4, N=6, R=100., mu=-1.-1.j, shift=True):
    
    m = VV.shape[0]
    
    
    lmin = m
    
    def isincontour(z):
        tolcont = 1e-8
        return abs(z-mu)<R-tolcont
    
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
                dA = (R/N)*VV*((phi-mu)**p)*exp(1j*t)
            else:
                dA = (R/N)*VV*(phi**p)*exp(1j*t)
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
    tolres = 1e-2
    vects = dot(V01,svects)
    
    if not failed:
        for a in range(k):
            if not isincontour(lambs[a]):
                failed = True
                print "Test: not in contour"
                break
            """
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
                    """
    
    if failed: # if it failed either of the two above checks then schur decompose
    
        try:
            U, Q, sdim = schur(D, output='complex', sort=isincontour) # schur decompose
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
            """
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
                    """
                
        if failed:
            lambs = delete(lambs,deletelist)
            vects = delete(vects,deletelist,1)
            print "Some values were incorrect, deleted"
        else:
            print "Values are correct"
    
    if lambs.shape[0] == 0:
        print "No eigenvalues computed within contour"
    else:
        """
        # error of solution; is T(lambda)v = 0?
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
        """
    
    
    print "Number of eigenvalues found: "+str(len(lambs))
    return lambs, vects
