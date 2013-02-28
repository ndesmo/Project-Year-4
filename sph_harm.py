import numpy as np
from scipy.special import sph_harm, sph_jn

N = 100

# Will be using spherical coordinates
theta = np.linspace(0, 2*np.pi, N)
phi = np.linspace(0, np.pi, N/2)
R = np.linspace(0, 1., N/2)

# convert into cartesian arrays
X = np.zeros(N**3/4) ; Y = np.zeros(N**3/4) ; Z = np.zeros(N**3/4) ; S = np.zeros(N**3/4)
i = 0
for r in R:
    for t in theta:
        for p in phi:
            X[i] = r*np.sin(t)*np.cos(p)
            Y[i] = r*np.sin(t)*np.sin(p)
            Z[i] = r*np.cos(t)
            i += 1

# Initial parameters for the eigenfunction        
m = 2
l = 0
k = 1.

# Generate values for eigenfunctions at each point
i = 0
for r in R:
    jm, jmdash = sph_jn(m, r)
    Jm = jm[m]
    for t in theta:
        for p in phi:
            Ym = sph_harm(l, m, t, p)
            S[i] = Jm*Ym
            i += 1

# Plot a 3D visualization
from mayavi.mlab import points3d

points3d(X, Y, Z, S)

        
    