import numpy as np
from scipy.special import sph_harm, sph_jn

""" Plot the eigenmodes of the sphere """

N = 30
m = 0
l = 0
k = 1.

X = np.linspace(-1., 1., N)
Y = X; Z = X
S = np.zeros((N,N,N))

i = 0
for x in X:
    j = 0
    for y in Y:
        k = 0
        for z in Z:
            r = x**2+y**2+z**2
            if r < 1.0:
                t = np.arccos(z/r)
                p = np.arctan(y/x)
                jm, jmdash = sph_jn(m, r)
                Jm = jm[l]
                Ym = sph_harm(l, m, t, p)
                S[i,j,k] = np.real(Jm*Ym)
            else:
                S[i,j,k] = -1.1
            k += 1
        j += 1
    i += 1
    
print np.min(S), np.max(S)


# Plot a 3D visualization
from mayavi import mlab
plot = mlab.figure(bgcolor = (1.0, 1.0, 1.0))
mlab.pipeline.volume(mlab.pipeline.scalar_field(S), vmin = -1.0, vmax = 1.0)
#mlab.points3d(XX, YY, ZZ, SS, scale_mode = 'none', opacity = 1.0, mode = 'point')
mlab.show()
        
    