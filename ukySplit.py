"""Produced at the University of Kentucky Markey Cancer Center
03/2019
by Brian Davis and Sally Ellingson
with funds from grant ####.

Example usage:

    from ukySplit import ukyDataSet
    import numpy as np
    import pandas as pd

    # Making artificial data:
    np.random.seed(42)
    X = np.random.sample((100,10))
    y = np.random.randint(2, size=100)
    attr_df = pd.DataFrame(np.array([hex(t) for t in range(100)]), columns=['smiles_col'])
    class Dset:
        def __init__(self, X, y):
            self.X = X
            self.y = y
    dataset = Dset(X,y)

    # Creating the ukyDataset, running optimizer, and splitting the data set:
    data = ukyDataSet(dataset.X, dataset.y, ids=attr_df['smiles_col'].values, Metric='euclidean')
    train_cv, test = data.splitData()


Example output:

    -- Generation 0 -- Time (hrs): 0.0002 -- Min score: -0.1299
    -- Mean score: -0.064 -- Unique Valid splits: 13/250 -- Var splits: 0.1599

"""

import warnings
import psutil
from time import time
import random
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin_min

from deap import base
from deap import creator
from deap import tools

safetyFactor = 3.75  # (3.75) Fraction of avaliable RAM to use for distance matrix computation
sizeBound = int(np.sqrt(psutil.virtual_memory().available / 8)/safetyFactor)
# sizeBound = 15100
"""sizeBound is the Max size of a dataset so that its distance matrix reliably
 fits in user's computer memory."""


def approx(array):
    return np.ceil(50*array)/51


class ukyDataSet:
    """A class for organizing protein/ligand datasets for optimal splitting.

    Inputs:
        Feature, label, and id (SMILES) arrays

    Parameters:
        targetRatio -(0.8)- the preferred training/datasetSize ratio.
        ratioTol -(0.01)- the percentage difference allowed between a split ratio and the target ratio.
        balanceTol -(0.05)- the percentage difference allowed between the active/decoy ratio in the total dataset
                            and the same ratio in the validation set.
        atomwise -(True)- Use the Atomwise AVE bias (True), or the new (more robust) bias score (to appear).
        Metric -('jaccard')- The sklearn metric to use for the feature distances.
     """

    def __init__(self, features, labels, ids, targetRatio=0.8, ratioTol=0.01,
                 balanceTol=0.05, atomwise=True, Metric='jaccard'):
        self.features = features
        self.labels = labels
        self.ids = ids
        self.targetRatio = targetRatio
        self.ratioTol = ratioTol
        self.balanceTol = balanceTol
        self.atomwise = atomwise
        self.metric = Metric
        self.size = len(ids)
        if self.size > sizeBound:
            self.isTooBig = True
        else:
            self.isTooBig = False
            with warnings.catch_warnings():
                # Suppress warning from distance matrix computation (int->bool)
                warnings.simplefilter("ignore")
                self.distanceMatrix = pairwise_distances(self.features, metric=Metric)

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
            validActive = self.features[(split == 0) & (self.labels == 1)]
            validDecoy = self.features[(split == 0) & (self.labels == 0)]
            trainActive = self.features[(split == 1) & (self.labels == 1)]
            trainDecoy = self.features[(split == 1) & (self.labels == 0)]
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

    def geneticOptimizer(self, numGens, printFreq=100, POPSIZE=250, TOURNSIZE=3,
                         CXPB=0.5, MUTPB=0.4, INDPB=0.075, scoreGoal=0.02):
        """
        A method for the genetic optimizer.

        Parameters:
            POPSIZE -(250)- Number of active individuals in a generation
            INDPB -(0.075)- Percent of individual bits randomly flipped
            TOURNSIZE -(3)- Size of selection tournaments
            CXPB -(0.5)- Probability with which two individuals are crossed
            MUTPB -(0.4)- Probability for mutating an individual
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
                print('-- Generation {}'.format(gen)
                      + ' -- Time (hrs): {}'.format(np.round((time() - t0)/3600, 4))
                      + ' -- Min score: {}'.format(np.round(minScore, 4))
                      + ' -- Mean score: {}'.format(np.round(meanScore, 4))
                      + ' -- Unique Valid splits: {}/{}'.format(numUnique, POPSIZE)
                      + ' -- Var splits: {}'.format(np.round(var, 4))
                      )
            gen += 1
        scores = [self.computeScore(split) for split in pop]
        return pop[np.argmin(scores)]

    def splitData(self):
        bigFrame = pd.DataFrame(self.features)
        bigFrame['labels'] = self.labels
        split = self.geneticOptimizer(numGens=100, printFreq=50, POPSIZE=250, TOURNSIZE=3,
                                      CXPB=0.5, MUTPB=0.4, INDPB=0.075, scoreGoal=0.02)
        bigFrame['split'] = split
        bigFrame['ids'] = self.ids
        return bigFrame[split == 1].drop('split', axis=1), bigFrame[split == 0].drop('split', axis=1)

