import numpy as np
import time
import scipy.stats as st

''' A module for storing useful functions for analyzing AVE bias.'''

###############################################################################
#Performance metric

def performance(min_score, mean, std, num_samples):
    cdf = st.norm.cdf(min_score,
                      loc=mean,
                      scale=std)
    return -np.log(num_samples*cdf)

###############################################################################

class data_set:
    '''A class for a labeled data set which allows
    easy computation of the AVE bias.'''
    def __init__(self, distance_matrix, Labels):
        self.distanceMatrix = distance_matrix
        self.labels = Labels
        self.size = len(Labels)
        self.bias_samples = []
        self.splits = []
        self.times = []
        self.comp_time = 0

    def computeAVEbias(self, split):
        minPosPosDist = np.amin(self.distanceMatrix[(split==0) & (self.labels==1),:]\
                                       [:,(split==1) & (self.labels==1)],
                                       axis=1)
        minPosNegDist = np.amin(self.distanceMatrix[(split==0) & (self.labels==1),:]\
                                   [:,(split==1) & (self.labels==0)],
                                   axis=1)

        minNegPosDist = np.amin(self.distanceMatrix[(split==0) & (self.labels==0),:]\
                                   [:,(split==1) & (self.labels==1)],
                                   axis=1)
        minNegNegDist = np.amin(self.distanceMatrix[(split==0) & (self.labels==0),:]\
                                   [:,(split==1) & (self.labels==0)],
                                   axis=1)
        bias = np.mean(minPosNegDist) + np.mean(minNegPosDist) \
                - np.mean(minPosPosDist) - np.mean(minNegNegDist)
        return bias

    def randSplit(self, q=0.8):
        '''Produce a random training / validation split of the data
        with the probability of training being q'''
        split =  np.random.choice( 2, size=self.size, p=[1-q , q] )
        return split

    def sample(self, duration, q=0.8):
        '''Produce an array of sampled biases
        together with the times required to compute them.
        Initializes the standard deviation, mean, compTimes.'''
        compTime = 0
        while compTime < duration:
            t0 = time.time()
            split = self.randSplit(q)
            bias = self.computeAVEbias(split)
            if not np.isnan(bias):
                self.splits.append(list(split))
                self.bias_samples.append(bias)
                self.times.append(time.time() - t0)
                compTime += time.time() - t0

###############################################################################
