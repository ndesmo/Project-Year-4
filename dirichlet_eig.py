#!/usr/bin/env python

# Help Python find the bempp module
import sys
sys.path.append("/home/nick/bempp/python")

from bempp.lib import *
import numpy as np
import scipy
from solveeig import eigSolve

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

# Construct elementary operators        


slpOp = createHelmholtz3dSingleLayerBoundaryOperator(
    context, pwiseConstants, pwiseLinears, pwiseConstants,2)


def applySingleLayerInverse(k,V):
    """Return V(k)^{-1}V, where V is a given matrix"""

    n = pwiseConstants.globalDofCount()
    slpOp = createHelmholtz3dSingleLayerBoundaryOperator(
        context, pwiseConstants, pwiseLinears, pwiseConstants,k)

    slpWeak = slpOp.weakForm().asMatrix()
    return scipy.linalg.solve(slpWeak,V)

Vhat = np.random.rand(2,2)
k = 4
VV = applySingleLayerInverse(k,Vhat)

lambs, vects = eigSolve(VV)
print lambs
print vects