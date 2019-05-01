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
seed = 42  # random seed used in optimization

sizeBound = 15100
"""sizeBound: max size of dataset that reliably
 fits distance matrix in my computer's memory. -- Brian"""


def approx(scalar):
    return int(50*scalar)


Approx = np.vectorize(approx)


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
        ratioTol (0.01) Allowable deviation from targetRatio
        balanceTol (0.02) Allowable deviation from balance ratio
    """

    def __init__(self, target_id, activeFile, decoyFile, targetRatio=0.8, ratioTol=0.01,
                 balanceTol=0.02, atomwise=True, Metric='jaccard'):
        # Gathering fingerprints
        print('Gathering fingerprints')
        decoyPrints = makePrints(decoyFile).drop_duplicates()
        activePrints = makePrints(activeFile).drop_duplicates()
        # Adding label columns
        activePrints['Labels'] = int(1)
        decoyPrints['Labels'] = int(0)
        # Combining into one dataframe
        fPrints = activePrints.append(decoyPrints, ignore_index=True)
        # Creating useful instance variables
        self.target = target_id
        self.size = fPrints.shape[0]
        self.fingerprints = fPrints.drop('Labels', axis=1)
        if self.size > sizeBound:
            self.isTooBig = True
            print('Is too big for distance Matrix')
        else:
            self.isTooBig = False
            # Store distance matrix if not too big
            with warnings.catch_warnings():
                # Suppress warning from distance matrix computation (float->bool)
                warnings.simplefilter("ignore")
                print('Computing distance matrix')
                self.distanceMatrix = pairwise_distances(self.fingerprints, metric=Metric)
        self.labels = fPrints['Labels']
        self.targetRatio = targetRatio
        self.ratioTol = ratioTol
        self.balanceTol = balanceTol
        self.atomwise = atomwise
        self.metric = Metric
        # Initialize the optimal split and its score
        self.optRecord = []
        self.bestScore = 2.0
        self.bestSplit = np.zeros(self.size)
        print('Finishing initialization')

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

    def computeScores(self, split, check=True):
        if check:
            if not self.validSplit(split):
                return 1.0, 1.0
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
                """AA - AI"""
                activeMeanDistance = np.mean(Approx(actDecoyDistances[1]) - Approx(actActDistances[1]))/51
                """II - IA"""
                decoyMeanDistance = np.mean(Approx(decoyActDistances[1]) - Approx(decoyDecoyDistances[1]))/51
            else:
                activeMeanDistance = np.mean(actDecoyDistances[1] - actActDistances[1])
                decoyMeanDistance = np.mean(decoyActDistances[1] - decoyDecoyDistances[1])
            return activeMeanDistance, decoyMeanDistance
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
                scores = np.mean(Approx(minPosNegDist) - Approx(minPosPosDist))/51,\
                         np.mean(Approx(minNegPosDist) - Approx(minNegNegDist))/51
            else:
                scores = np.mean(minPosNegDist - minPosPosDist),\
                         np.mean(minNegPosDist - minNegNegDist)
            return scores[0], scores[1]

    def objectiveFunction(self, split):
        x = self.computeScores(split)[0]
        y = self.computeScores(split)[1]
        if self.atomwise:
            return x + y,
        else:
            return np.sqrt(x**2 + y**2),

    def randSplit(self):
        """
        Produce a random training / validation split of the data
        with the probability of training being targetRatio
        """
        valid = False
        while not valid:
            split = np.random.choice(2, size=self.size, p=[1 - self.targetRatio, self.targetRatio])
            valid = self.validSplit(split)
        return split

    def sample(self, numSamples):
        np.random.seed()
        sample = []
        for i in range(numSamples):
            newSplit = self.randSplit()
            newScore = self.objectiveFunction(newSplit)[0]
            sample.append(newScore)
        return sample

    def geneticOptimizer(self, numGens, printFreq=100, POPSIZE=1000, TOURNSIZE=4,
                         CXPB=0.18, MUTPB=0.39, INDPB=0.005, onePoint=True, scoreGoal=0.01, verbose=False):
        """
        A method for the genetic optimizer.

        Parameters:
            POPSIZE = 1000 #(1000) Number of individuals in a generation
            INDPB = 0.075 #(0.075) Percent of individual bits randomly flipped
            TOURNSIZE = 4 #(4) Size of selection tournaments
            CXPB = 0.5 #(0.5) Probability with which two individuals are crossed
            MUTPB = 0.4 #(0.4) Probability for mutating an individual
            verbose = False  #(False) Print more statistics
        Taken (with minor changes) from example code:
        https://deap.readthedocs.io/en/master/examples/ga_onemax.html"""

        print(f'{self.target} - Initializing optimizer ...')
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
            toolbox.register("evaluate", self.objectiveFunction)
            if onePoint:
                toolbox.register("mate", tools.cxOnePoint)
            else:
                toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutFlipBit, indpb=INDPB)
            toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)
        # Set random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        pop = toolbox.population(n=POPSIZE)
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        gen = 0
        minScore = 2.0
        print(f'{self.target} - Beginning optimization...')
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
                else:
                    scores = validPop.apply(lambda x: self.objectiveFunction(x)[0], axis=1)
                    meanScore = np.mean(scores.values)
                    minScore = np.min(scores.values)
                    optSplit = validPop[scores == minScore]
                    scoreParts = self.computeScores(optSplit.values[0])
                if verbose:
                    print(f'{self.target} -- Generation {gen}'
                          + ' -- Time (sec): {}'.format(np.round((time() - t0), 2))
                          + ' -- Min score: {}'.format(np.round(minScore, 4))
                          + f' -- Score parts: {scoreParts[0]}, {scoreParts[1]}'
                          + ' -- Mean score: {}'.format(np.round(meanScore, 4))
                          + ' -- Unique Valid splits: {}/{}'.format(numUnique, POPSIZE)
                          )
                else:
                    print(f'{self.target} -- Generation {gen}'
                          + ' -- Time (sec): {}'.format(np.round((time() - t0), 2))
                          + ' -- Min score: {}'.format(np.round(minScore, 4))
                          + f' -- Score parts: {scoreParts[0]}, {scoreParts[1]}'
                          )
                self.optRecord.append((time() - t0, scoreParts[0], scoreParts[1], minScore))
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
