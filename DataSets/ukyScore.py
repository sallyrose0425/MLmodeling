import numpy as np
import pandas as pd
import random
from time import time
import warnings
import gzip
import psutil

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.metrics import pairwise_distances

from sklearn.metrics.pairwise import pairwise_distances_argmin_min

from deap import base
from deap import creator
from deap import tools

''' A module for storing useful functions for analyzing split score.'''


###############################################################################

safetyFactor = 3  # (3) Fraction of avaliable RAM to use for distance matrix computation
mem = psutil.virtual_memory()
sizeBound = int(np.sqrt(mem.available / 8)/safetyFactor)
# sizeBound = 15100
"""sizeBound: max size of dataset that reliably
 fits distance matrix in user's computer's memory."""


def approx(array):
    return np.ceil(50*array)/51


def finger(mol):
    fprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    return list(fprint)


def makePrints(s):
    try:
        inf = gzip.open(s)
        gzsuppl = Chem.ForwardSDMolSupplier(inf)
        mols = [x for x in gzsuppl if x is not None]
        prints = [finger(mol) for mol in mols]
        prints = pd.DataFrame(prints).dropna()
        return prints
    except:
        print('Unable to open...')
        return


class data_set:
    """
    A class for a labeled data set which allows
    easy computation of the split score.

    Inputs:

    Parameters:
        targetRatio (0.8) Target percentage of data used for training
        ratioTol (0.01) Allowable deviation from SPLIT ratio
        balanceTol (0.05) Allowable deviation from balance ratio
    """

    def __init__(self, activeFile, decoyFile, targetRatio=0.8, ratioTol=0.01,
                 balanceTol=0.05, atomwise=False, Metric='jaccard'):
        decoyPrints = makePrints(decoyFile)
        activePrints = makePrints(activeFile)
        activePrints['Labels'] = int(1)
        decoyPrints['Labels'] = int(0)
        fPrints = activePrints.append(decoyPrints, ignore_index=True)
        self.size = fPrints.shape[0]
        self.fingerprints = fPrints.drop('Labels', axis=1)
        if self.size > sizeBound:
            self.isTooBig = True
        else:
            self.isTooBig = False
            with warnings.catch_warnings():
                # Suppress warning from distance matrix computation (float->bool)
                warnings.simplefilter("ignore")
                self.distanceMatrix = pairwise_distances(self.fingerprints, metric=Metric)
        self.labels = fPrints['Labels']
        self.targetRatio = targetRatio
        self.ratioTol = ratioTol
        self.balanceTol = balanceTol
        self.atomwise = atomwise
        self.metric = Metric
        self.optRecord = []

    def validSplit(self, split):
        """
        Return True if the split has the proper training/validation ratio
        for both actives and decoys
        """

        numTraining = np.sum(split)
        numValidation = self.size - numTraining
        trueRatio = float(numTraining) / self.size
        ratioError = np.abs(self.targetRatio - trueRatio)
        numActives = np.sum(self.labels)
        balance = float(numActives) / self.size
        numActiveTraining = np.sum(split & self.labels)
        numActiveValidation = numActives - numActiveTraining
        trueValidationBalance = float(numActiveValidation) / numValidation
        balanceError = np.abs(trueValidationBalance - balance)
        check = (balanceError < self.balanceTol) and (ratioError < self.ratioTol)
        nonEmpty = bool(numActiveValidation) and bool(numValidation - numActiveValidation)
        return check and nonEmpty

    def computeScore(self, split):
        if not self.validSplit(split):
            return 2.0,
        if self.isTooBig:
            validActive = self.fingerprints[(split == 0) & (self.labels == 1)]
            validDecoy = self.fingerprints[(split == 0) & (self.labels == 0)]
            trainActive = self.fingerprints[(split == 1) & (self.labels == 1)]
            trainDecoy = self.fingerprints[(split == 1) & (self.labels == 0)]
            actActDistances = pairwise_distances_argmin_min(validActive, trainActive, metric=self.metric)
            actDecoyDistances = pairwise_distances_argmin_min(validActive, trainDecoy, metric=self.metric)
            decoyActDistances = pairwise_distances_argmin_min(validDecoy, trainActive, metric=self.metric)
            decoyDecoyDistances = pairwise_distances_argmin_min(validDecoy, trainDecoy, metric=self.metric)
            if self.atomwise:
                activeMeanDistance = np.mean(approx(actDecoyDistances[1]) - approx(actActDistances[1]))
                decoyMeanDistance = np.mean(approx(decoyActDistances[1]) - approx(decoyDecoyDistances[1]))
            else:
                activeMeanDistance = np.mean(actDecoyDistances[1] - actActDistances[1])
                decoyMeanDistance = np.mean(decoyActDistances[1] - decoyDecoyDistances[1])
            return activeMeanDistance + decoyMeanDistance
        else:
            minPosPosDist = np.amin(
                self.distanceMatrix[(split == 0) & (self.labels == 1), :][:, (split == 1) & (self.labels == 1)], axis=1)
            minPosNegDist = np.amin(
                self.distanceMatrix[(split == 0) & (self.labels == 1), :][:, (split == 1) & (self.labels == 0)], axis=1)

            minNegPosDist = np.amin(
                self.distanceMatrix[(split == 0) & (self.labels == 0), :][:, (split == 1) & (self.labels == 1)], axis=1)
            minNegNegDist = np.amin(
                self.distanceMatrix[(split == 0) & (self.labels == 0), :][:, (split == 1) & (self.labels == 0)], axis=1)
            if self.atomwise:
                score = np.mean(approx(minPosNegDist)) + np.mean(approx(minNegPosDist))\
                        - np.mean(approx(minPosPosDist)) - np.mean(approx(minNegNegDist))
            else:
                score = np.mean(minPosNegDist) + np.mean(minNegPosDist)\
                        - np.mean(minPosPosDist) - np.mean(minNegNegDist)
            return score,

    def randSplit(self):
        """
        Produce a random training / validation split of the data
        with the probability of training being q
        """

        split = np.random.choice(2, size=self.size, p=[1 - self.targetRatio, self.targetRatio])
        return split

    def geneticOptimizer(self, numGens, printFreq=100, POPSIZE=1000, TOURNSIZE=3,
                         CXPB=0.4, MUTPB=0.4, INDPB=0.075, scoreGoal=0.02, verbose=False):
        """
        A method for the genetic optimizer.

        Parameters:
            POPSIZE = 1000 #(250) Number of active individuals in a generation
            INDPB = 0.075 #(0.05) Percent of individual bits randomly flipped
            TOURNSIZE = 3 #(3) Size of selection tournaments
            CXPB = 0.5 #(0.5) Probability with which two individuals are crossed
            MUTPB = 0.4 #(0.4) Probability for mutating an individual
            verbose = False  # Print more statistics
        """

        t0 = time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
            toolbox = base.Toolbox()
            toolbox.register("attr_bool", np.random.choice, 2, p=[1 - self.targetRatio, self.targetRatio])
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, self.size)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", self.computeScore)
            toolbox.register("mate", tools.cxOnePoint)
            toolbox.register("mutate", tools.mutFlipBit, indpb=INDPB)
            toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)
        np.random.seed(42)
        random.seed(42)
        pop = toolbox.population(n=POPSIZE)
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        gen = 0
        minScore = 2.0
        while gen < numGens and scoreGoal < minScore:
            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            pop[:] = offspring
            if gen % printFreq == 0:
                Pop = pd.DataFrame(pop)
                validPop = Pop[Pop.apply(lambda x: self.validSplit(x), axis=1)]
                validPop = validPop.drop_duplicates()
                numUnique = len(validPop)
                if numUnique == 0:
                    meanScore = np.nan
                    minScore = np.nan
                    var = np.nan
                else:
                    scores = validPop.apply(lambda x: self.computeScore(x)[0], axis=1)
                    meanScore = np.mean(scores.values)
                    minScore = np.min(scores.values)
                    if numUnique == 1:
                        var = 0.0
                    else:
                        var = validPop.var().mean()
                if verbose:
                    print('-- Generation {}'.format(gen)
                          + ' -- Time (sec): {}'.format(np.round((time() - t0), 2))
                          + ' -- Min score: {}'.format(np.round(minScore, 4))
                          + ' -- Mean score: {}'.format(np.round(meanScore, 4))
                          + ' -- Unique Valid splits: {}/{}'.format(numUnique, POPSIZE)
                          + ' -- Var splits: {}'.format(np.round(var, 4))
                          )
                else:
                    print('-- Generation {}'.format(gen)
                          + ' -- Time (sec): {}'.format(np.round((time() - t0), 2))
                          + ' -- Min score: {}'.format(np.round(minScore, 4))
                          )
                self.optRecord.append((time() - t0, minScore))
            gen += 1
        return pop

    def weights(self, split):
        if self.isTooBig:
            with warnings.catch_warnings():
                # Suppress warning from distance matrix computation (float->bool)
                warnings.simplefilter("ignore")
                validActive = self.fingerprints[(split == 0) & (self.labels == 1)]
                validDecoy = self.fingerprints[(split == 0) & (self.labels == 0)]
                trainActive = self.fingerprints[(split == 1) & (self.labels == 1)]
                trainDecoy = self.fingerprints[(split == 1) & (self.labels == 0)]
            actActDistances = pairwise_distances_argmin_min(validActive, trainActive, metric=self.metric)
            actDecDistances = pairwise_distances_argmin_min(validActive, trainDecoy, metric=self.metric)
            decActDistances = pairwise_distances_argmin_min(validDecoy, trainActive, metric=self.metric)
            decDecDistances = pairwise_distances_argmin_min(validDecoy, trainDecoy, metric=self.metric)
        else:
            actActDistances = self.distanceMatrix[(split == 0) & (self.labels == 1), :][:,
                              (split == 1) & (self.labels == 1)]
            actDecDistances = self.distanceMatrix[(split == 0) & (self.labels == 1), :][:,
                              (split == 1) & (self.labels == 0)]
            decActDistances = self.distanceMatrix[(split == 0) & (self.labels == 0), :][:,
                              (split == 1) & (self.labels == 1)]
            decDecDistances = self.distanceMatrix[(split == 0) & (self.labels == 0), :][:,
                              (split == 1) & (self.labels == 0)]
        decWeights = np.amin(decDecDistances, axis=1) / np.amin(decActDistances, axis=1)
        actWeights = np.amin(actActDistances, axis=1) / np.amin(actDecDistances, axis=1)
        holdWeights = np.zeros(self.size)
        validActiveIndices = np.where((split == 0) & (self.labels == 1))[0]
        for i in range(len(validActiveIndices)):
            holdWeights[validActiveIndices[i]] = actWeights[i]

        validDecoyIndices = np.where((split == 0) & (self.labels == 0))[0]
        for i in range(len(validDecoyIndices)):
            holdWeights[validDecoyIndices[i]] = decWeights[i]
        return holdWeights


"""
cd DataSets

from importlib import reload
import os
import ukyScore
import numpy as np
dataset = 'dekois'
target_id = 'ADRB2'
prefix = os.getcwd() + '/' + dataset + '/'
activeFile = prefix + 'ligands/' + target_id + '.sdf.gz'
decoyFile = prefix + 'decoys/' + target_id + '_Celling-v1.12_decoyset.sdf.gz'

reload(ukyScore)

data = ukyScore.data_set(activeFile, decoyFile, balanceTol=0.01)

for tournsize in np.linspace(.375, .425, num=5):
    splits = data.geneticOptimizer(numGens=1000, printFreq=100, POPSIZE=1000, TOURNSIZE=9,
                             CXPB=0.0, MUTPB=0.8, INDPB=0.075, scoreGoal=0.02)
    print(data.optRecord)
    
scores = [data.computeScore(split) for split in splits]
split = np.array(splits[np.argmin(scores)])

predictions = data.nearestNeighborPredictions(split)
perf = data.weightedPerformance(split, predictions, a=1, b=0.5)
perf
"""

"""
labels = data.labels.values
activeValidWeights = np.multiply(holdWeights, labels)
decoyValidWeights = np.multiply(holdWeights, 1 - labels)
activeTotalWeight = np.sum(activeValidWeights)
decoyTotalWeight = np.sum(decoyValidWeights)
truePositiveWeight = np.sum(np.multiply(activeValidWeights[split == 0], predictions))
trueNegativeWeight = np.sum(np.multiply(decoyValidWeights[split == 0], 1 - predictions))
"""

"""
    def nearestNeighborPredictions(self, split):
        trainingLabels = self.labels[split == 1].reset_index(drop=True)
        distancesToTraining = pd.DataFrame(self.distanceMatrix[split == 0, :][:, split == 1])
        closest = distancesToTraining.idxmin(axis=1)
        nearestNeighbors = np.array([trainingLabels[x] for x in closest])
        return nearestNeighbors
    def sample(self, numSamples):
        Produce an array of sampled biases
        together with the times required to compute them.
        Initializes the standard deviation, mean, compTimes

        np.random.seed(42)
        t0 = time()
        s = 0
        while s < numSamples:
            split = self.randSplit()
            if self.validSplit(split):
                score = self.computeScore(split)
                if not np.isnan(score):
                    self.splits.append(list(split))
                    self.score_samples.append(score)
                    s += 1
        self.comp_time += time() - t0
    def splitData(self, split):
        self.trainingFeatures = self.fingerprints[split == 1]
        self.validationFeatures = self.fingerprints[split == 0]
        self.trainingLabels = self.labels[split == 1]
        self.validationLabels = self.labels[split == 0]
"""