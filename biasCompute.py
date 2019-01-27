

from numba import njit
import numpy as np


@njit(parallel=True)
def getGammaValue(index, posArray, negArray):
    '''index corresponding to v, lists of elements to check minimum over'''
    posRecord = 1
    for j in posArray:
        checkValue = distanceMatrix_tmp[(index,j)]
        if checkValue < posRecord:
            posRecord = checkValue
    posVal = int(n*posRecord)
    #
    negRecord = 1
    for k in negArray:
        checkValue = distanceMatrix_tmp[(index,k)]
        if checkValue < negRecord:
            negRecord = checkValue
    negVal = int(n*negRecord)
    return  posVal - negVal 

@njit(parallel=True)
def aveBias(validPos_List,
                validNeg_List,
                trainPos_List,
                trainNeg_List):
    posSum = 0
    for i in validPos_List:
        posSum += getGammaValue(i , trainPos_List , trainNeg_List)
    #
    negSum = 0
    for i in validNeg_List:
        posSum += getGammaValue(i , trainPos_List , trainNeg_List)
    #
    invPos = np.reciprocal(float(len(validPos_List)))
    invNeg = np.reciprocal(float(len(validNeg_List)))
    result = np.reciprocal(float(n+1))*(invPos*posSum + invNeg*negSum)
    return np.round(result,3)

###############################################################################
    
n = 100

numValid = 200
numTrain = 800

numPos = 500
numNeg = numTrain + numValid - numPos

p = 10**2
validPosList = np.array(range(0,1*p))
validNegList = np.array(range(1*p,2*p))
trainPosList = np.array(range(2*p,6*p))
trainNegList = np.array(range(6*p,10*p))
    
distanceMatrix_tmp = np.random.randint(n+1, size=(10*p, 10*p))


aveBias(validPosList,
      validNegList,
      trainPosList,
      trainNegList)
        

