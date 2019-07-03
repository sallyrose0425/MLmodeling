"""
Produced at the University of Kentucky Markey Cancer Center
03/2019
by Brian Davis and Sally Ellingson
with funds from grant ####.

Example usage:

"""

import warnings
from time import time
import random
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.metrics import pairwise_distances
from deap import base
from deap import creator
from deap import tools


class ukyDataSet:
    """A class for organizing protein/ligand datasets for optimal splitting."""

    def __init__(self, features, labels, targetRatio=0.8, ratioTol=0.01, balanceTol=0.05, AVE=True):
        self.features = features.astype(bool)
        self.labels = labels
        self.targetRatio = targetRatio
        self.ratioTol = ratioTol
        self.balanceTol = balanceTol
        self.AVE = AVE
        self.size = len(labels)
        if self.size > 15100:
            self.distanceMatrix = None
        else:
            self.distanceMatrix = pairwise_distances(self.features, metric='jaccard')

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
        else:
            if self.distanceMatrix is None:
                validActive = self.features[(split == 0) & (self.labels == 1)]
                validDecoy = self.features[(split == 0) & (self.labels == 0)]
                trainActive = self.features[(split == 1) & (self.labels == 1)]
                trainDecoy = self.features[(split == 1) & (self.labels == 0)]
                actActDistances = pairwise_distances_argmin_min(validActive, trainActive, metric='jaccard')[1]
                actDecoyDistances = pairwise_distances_argmin_min(validActive, trainDecoy, metric='jaccard')[1]
                decoyActDistances = pairwise_distances_argmin_min(validDecoy, trainActive, metric='jaccard')[1]
                decoyDecoyDistances = pairwise_distances_argmin_min(validDecoy, trainDecoy, metric='jaccard')[1]
            else:
                actActDistances = np.amin(
                    self.distanceMatrix[(split == 0) & (self.labels == 1), :][:, (split == 1) & (self.labels == 1)],
                    axis=1)
                actDecoyDistances = np.amin(
                    self.distanceMatrix[(split == 0) & (self.labels == 1), :][:, (split == 1) & (self.labels == 0)],
                    axis=1)
                decoyActDistances = np.amin(
                    self.distanceMatrix[(split == 0) & (self.labels == 0), :][:, (split == 1) & (self.labels == 1)],
                    axis=1)
                decoyDecoyDistances = np.amin(
                    self.distanceMatrix[(split == 0) & (self.labels == 0), :][:, (split == 1) & (self.labels == 0)],
                    axis=1)
            activeMeanDistance = np.mean(actDecoyDistances - actActDistances)
            decoyMeanDistance = np.mean(decoyActDistances - decoyDecoyDistances)
            if self.AVE:
                score = activeMeanDistance + decoyMeanDistance
            else:
                score = np.sqrt(activeMeanDistance**2 + decoyMeanDistance**2)
            return score,

    def geneticOptimizer(self, numGens, popsize=250):
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
            # Create optimizer tools
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
            toolbox = base.Toolbox()
            toolbox.register("attr_bool", np.random.choice, 2, p=[1 - self.targetRatio, self.targetRatio])
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, self.size)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", self.computeScore)
            toolbox.register("mate", tools.cxOnePoint)
            toolbox.register("mutate", tools.mutFlipBit, indpb=0.075)
            toolbox.register("select", tools.selTournament, tournsize=3)
        np.random.seed(42)
        random.seed(42)
        pop = toolbox.population(n=popsize)
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        gen = 0
        minScore = 2.0
        while gen < numGens and 0.02 < minScore:
            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < 0.4:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            pop[:] = offspring
            if gen % 100 == 0:
                scores = [self.computeScore(split) for split in pop]
                print('-- Generation {}'.format(gen)
                      + ' -- Time (hrs): {}'.format(np.round((time() - t0)/3600, 4))
                      + ' -- Min score: {}'.format(np.round(np.min(scores), 4))
                      )
            gen += 1
        scores = [self.computeScore(split) for split in pop]
        opt_split = pop[np.argmin(scores)]
        # training_indices = np.where(opt_split)[0]
        # validation_indices = np.where(not opt_split)[0]
        # return training_indices, validation_indices
        return np.where(opt_split)[0], np.where(~opt_split.astype(bool))[0]

    def weights(self, split):
        if self.distanceMatrix is None:
            validActive = self.features[(split == 0) & (self.labels == 1)].astype(bool)
            validDecoy = self.features[(split == 0) & (self.labels == 0)].astype(bool)
            trainActive = self.features[(split == 1) & (self.labels == 1)].astype(bool)
            trainDecoy = self.features[(split == 1) & (self.labels == 0)].astype(bool)
            actActDistances = pairwise_distances_argmin_min(validActive, trainActive, metric='jaccard')
            actDecDistances = pairwise_distances_argmin_min(validActive, trainDecoy, metric='jaccard')
            decActDistances = pairwise_distances_argmin_min(validDecoy, trainActive, metric='jaccard')
            decDecDistances = pairwise_distances_argmin_min(validDecoy, trainDecoy, metric='jaccard')
            decWeights = decDecDistances / decActDistances
            actWeights = actActDistances / actDecDistances
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

    def get_validation_weights(self, training_indices, validation_indices):
        weights = self.weights(np.isin(np.arange(self.size), training_indices))
        sorted_weights = np.argsort(weights).tolist()
        validation_weights = [sorted_weights.index(index) for index in validation_indices]
        return (np.array(validation_weights) - len(training_indices)) / len(validation_indices)
