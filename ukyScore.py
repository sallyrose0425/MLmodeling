import numpy as np
import pandas as pd
import time

from sklearn.metrics.pairwise import pairwise_distances_argmin_min

''' A module for storing useful functions for analyzing split score.'''

###############################################################################

class data_set:
    '''A class for a labeled data set which allows
    easy computation of the split score.'''
	
    def __init__(self, distance_matrix, fPrints):
        self.fingerprints = fPrints.drop('Labels', axis=1)
        self.distanceMatrix = distance_matrix.values
        self.labels = fPrints['Labels']
        self.size = len(self.labels)
        self.bias_samples = []
        self.splits = []
        self.score_samples = []
        self.times = []
        self.comp_time = 0
        if np.shape(distance_matrix)[0] == 0:
            self.isTooBig = True
        else:
            self.isTooBig = False
        
		
    def computeScore(self, split):
        if self.isTooBig == True:
            validActive = self.fingerprints[(split==0) & (self.labels==1)]
            validDecoy = self.fingerprints[(split==0) & (self.labels==0)]
            trainActive = self.fingerprints[(split==1) & (self.labels==1)]
            trainDecoy = self.fingerprints[(split==1) & (self.labels==0)]
            actActDistances = pairwise_distances_argmin_min(validActive,
                                                   trainActive,
                                                   metric='jaccard')
            actDecoyDistances = pairwise_distances_argmin_min(validActive,
                                                   trainDecoy,
                                                   metric='jaccard')
            activeMeanDistance = np.mean(actDecoyDistances[1]
                                        - actActDistances[1]
                                        )
            decoyActDistances = pairwise_distances_argmin_min(validDecoy,
                                                   trainActive,
                                                   metric='jaccard')
            decoyDecoyDistances = pairwise_distances_argmin_min(validDecoy,
                                                   trainDecoy,
                                                   metric='jaccard')
            decoyMeanDistance = np.mean(decoyActDistances[1]
                                        - decoyDecoyDistances[1]
                                        )
            return activeMeanDistance + decoyMeanDistance
        else:
            minPosPosDist = np.amin(
                    self.distanceMatrix[(split==0) & (self.labels==1),:]\
                                           [:,(split==1) & (self.labels==1)],
                                           axis=1
                                    )
            minPosNegDist = np.amin(
                    self.distanceMatrix[(split==0) & (self.labels==1),:]\
                                       [:,(split==1) & (self.labels==0)],
                                       axis=1
                                    )
    
            minNegPosDist = np.amin(
                    self.distanceMatrix[(split==0) & (self.labels==0),:]\
                                       [:,(split==1) & (self.labels==1)],
                                       axis=1
                                    )
            minNegNegDist = np.amin(
                    self.distanceMatrix[(split==0) & (self.labels==0),:]\
                                       [:,(split==1) & (self.labels==0)],
                                       axis=1
                                    )
            score = np.mean(minPosNegDist) + np.mean(minNegPosDist) \
                    - np.mean(minPosPosDist) - np.mean(minNegNegDist)
            return score


    def randSplit(self, q=0.8):
        '''Produce a random training / validation split of the data
        with the probability of training being q'''
        split =  np.random.choice( 2, size=self.size, p=[1-q , q] )
        return split


    def validSplit(self,
                   split,
                   targetRatio=0.8,
                   ratioTol = 0.01,
                   balanceTol = 0.05
                   ):
        '''return True if the split has the proper training/validation ratio
        for both actives and decoys
        '''
        numTraining = np.sum(split)
        trueRatio = float(numTraining) / self.size
        ratioError = np.abs(targetRatio - trueRatio)
        numActives = np.sum(self.labels)
        balance = float(numActives)/self.size
        numActiveTraining = np.sum(split & self.labels)
        numActiveValidation = numActives - numActiveTraining
        trueValidationBalance = float(numActiveValidation) / (self.size - numTraining)
        balanceError = np.abs(trueValidationBalance - balance)
        return (balanceError < balanceTol) and (ratioError < ratioTol)
    
	
    def sample(self,
               numSamples,
               targetRatio=0.8,
               ratioTol = 0.1,
               balanceTol = 0.1,
               q=0.8
               ):
        '''Produce an array of sampled biases
        together with the times required to compute them.
        Initializes the standard deviation, mean, compTimes.'''
        t0 = time.time()
        s=0
        while s < numSamples:
            split = self.randSplit(q)
            if self.validSplit(split, targetRatio, ratioTol, balanceTol):
                score = self.computeScore(split)
                if not np.isnan(score):
                    self.splits.append(list(split))
                    self.score_samples.append(score)
                    s += 1
        self.comp_time += time.time() - t0

        
    def nearestNeighborPredictions(self, split):
        trainingLabels = self.labels[split==1].reset_index(drop=True)
        distancesToTraining = pd.DataFrame(self.distanceMatrix[split==0, :][:, split==1])
        closest = distancesToTraining.idxmin(axis=1)
        nearestNeighbors = np.array([trainingLabels[x] for x in closest])
        return nearestNeighbors


    def splitData(self, split):
        self.trainingFeatures = self.fingerprints[split==1]
        self.validationFeatures = self.fingerprints[split==0]
        self.trainingLabels = self.labels[split==1]
        self.validationLabels = self.labels[split==0]                     

