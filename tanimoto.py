import numpy as np
from numba import jit, njit, prange
from sklearn.base import BaseEstimator, TransformerMixin

@njit(fastmath=True)
def tanimotoDistance(fprint1, fprint2):
    totalBits = len(fprint1)
    assert len(fprint2) == totalBits , 'Number of bits do not agree.'
    commonBits = 0
    activeBits = 0
    for i in range(totalBits):
        commonBits += fprint1[i] * fprint2[i]
        activeBits += fprint1[i] + fprint2[i]
    return 1 - float(commonBits)/(activeBits - commonBits)
    
    
'''
Ex:    
fprint1 = np.array([0,1,1,0])
fprint2 = np.array([0,1,0,1])

tanimoto(fprint1, fprint2)
0.3333333333333333
'''

@jit(parallel=True)
def makeSimArray(U, V, W):
    '''Shape of array W is row(U) x row(V)'''
    dim1 = int(np.shape(U)[0])
    dim2 = int(np.shape(V)[0])
    for i in range(dim1):
        for j in prange(dim2):
            W[i,j] = float(tanimotoDistance(U[i], V[j]))
    return W   
    

###############################################################################

class similarityFeatures(BaseEstimator, TransformerMixin):
    '''A class for transforming fingerprints into a similarity matrix'''

    def __init__(self):
        pass
        
    def fit(self, V):
        self.fingerprints = V
        self.dim = np.shape(V)[0]
        
    def transform(self, U):
        numRows = np.shape(U)[0]
        W = np.zeros((numRows, self.dim))
        return makeSimArray(U, self.fingerprints, W)

'''
Usage:
feat = similarityFeatures()
feat.fit(U)
L = feat.transform(V)
'''
