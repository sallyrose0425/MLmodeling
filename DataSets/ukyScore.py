import numpy as np
import pandas as pd
import random
import time
import warnings

from sklearn.metrics.pairwise import pairwise_distances_argmin_min

from deap import base
from deap import creator
from deap import tools

''' A module for storing useful functions for analyzing split score.'''


###############################################################################


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

    def __init__(self, distance_matrix, fPrints, targetRatio=0.8, ratioTol=0.01, balanceTol=0.05):
        self.fingerprints = fPrints.drop('Labels', axis=1)
        self.distanceMatrix = distance_matrix
        self.labels = fPrints['Labels']
        self.size = len(self.labels)
        self.bias_samples = []
        self.splits = []
        self.score_samples = []
        self.times = []
        self.comp_time = 0
        self.targetRatio = targetRatio
        self.ratioTol = ratioTol
        self.balanceTol = balanceTol
        if np.shape(distance_matrix)[0] == 0:
            self.isTooBig = True
        else:
            self.isTooBig = False

    def validSplit(self, split):
        """
        Return True if the split has the proper training/validation ratio
        for both actives and decoys
        """
        numTraining = np.sum(split)
        trueRatio = float(numTraining) / self.size
        ratioError = np.abs(self.targetRatio - trueRatio)
        numActives = np.sum(self.labels)
        balance = float(numActives) / self.size
        numActiveTraining = np.sum(split & self.labels)
        numActiveValidation = numActives - numActiveTraining
        trueValidationBalance = float(numActiveValidation) / (self.size - numTraining)
        balanceError = np.abs(trueValidationBalance - balance)
        return (balanceError < self.balanceTol) and (ratioError < self.ratioTol)

    def computeScore(self, split, Metric='jaccard'):
        if not self.validSplit(split, self.targetRatio, self.ratioTol, self.balanceTol):
            return 2.0,
        if self.isTooBig:
            validActive = self.fingerprints[(split == 0) & (self.labels == 1)]
            validDecoy = self.fingerprints[(split == 0) & (self.labels == 0)]
            trainActive = self.fingerprints[(split == 1) & (self.labels == 1)]
            trainDecoy = self.fingerprints[(split == 1) & (self.labels == 0)]
            actActDistances = pairwise_distances_argmin_min(validActive, trainActive, metric=Metric)
            actDecoyDistances = pairwise_distances_argmin_min(validActive, trainDecoy, metric=Metric)
            activeMeanDistance = np.mean(actDecoyDistances[1] - actActDistances[1])
            decoyActDistances = pairwise_distances_argmin_min(validDecoy, trainActive, metric=Metric)
            decoyDecoyDistances = pairwise_distances_argmin_min(validDecoy, trainDecoy, metric=Metric)
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
            score = np.mean(minPosNegDist) + np.mean(minNegPosDist) - np.mean(minPosPosDist) - np.mean(minNegNegDist)
            return score,

    def randSplit(self):
        """
        Produce a random training / validation split of the data
        with the probability of training being q
        """
        split = np.random.choice(2, size=self.size, p=[1 - self.targetRatio, self.targetRatio])
        return split

    def sample(self, numSamples):
        """
        Produce an array of sampled biases
        together with the times required to compute them.
        Initializes the standard deviation, mean, compTimes.
        """
        t0 = time.time()
        s = 0
        while s < numSamples:
            split = self.randSplit()
            if self.validSplit(split):
                score = self.computeScore(split)
                if not np.isnan(score):
                    self.splits.append(list(split))
                    self.score_samples.append(score)
                    s += 1
        self.comp_time += time.time() - t0

    def nearestNeighborPredictions(self, split):
        trainingLabels = self.labels[split == 1].reset_index(drop=True)
        distancesToTraining = pd.DataFrame(self.distanceMatrix[split == 0, :][:, split == 1])
        closest = distancesToTraining.idxmin(axis=1)
        nearestNeighbors = np.array([trainingLabels[x] for x in closest])
        return nearestNeighbors

    def splitData(self, split):
        self.trainingFeatures = self.fingerprints[split == 1]
        self.validationFeatures = self.fingerprints[split == 0]
        self.trainingLabels = self.labels[split == 1]
        self.validationLabels = self.labels[split == 0]

    def geneticOptimizer(self, numGens, POPSIZE=250, TOURNSIZE=3, CXPB=0.5, MUTPB=0.4, INDPB=0.075):
        """
        A method for the genetic optimizer.

        Parameters:
            POPSIZE = 1000 #(250) Number of active individuals in a generation
            INDPB = 0.075 #(0.05) Percent of individual bits randomly flipped
            TOURNSIZE = 3 #(3) Size of selection tournaments
            CXPB = 0.5 #(0.5) Probability with which two individuals are crossed
            MUTPB = 0.4 #(0.2) Probability for mutating an individual
        """

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
        pop = toolbox.population(n=POPSIZE)
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        gen = 0
        while gen < numGens:
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
            gen += 1
        return pop
