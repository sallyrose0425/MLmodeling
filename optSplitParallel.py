import os
#import sys
from time import time
from glob import glob
import warnings
from multiprocessing import Pool
import random
import numpy as np
import pandas as pd

from deap import base
from deap import creator
from deap import tools

import ukyScore
###############################################################################
#Parameters
NUMGENS = 1000 #(100) Number of generations (unless stop criterion)
PRINTFREQ = int(NUMGENS/25) #(25) Frequency of print fitness

targetRatio = 0.8 #(0.8) Target percentage of data used for training
ratioTol = 0.01 #(0.005) Allowable deviation from SPLIT ratio
balanceTol = 0.05

INDPB = 0.075 #(0.05) Percent of individual bits randomly flipped
TOURNSIZE = 3 #(3) Size of selection tournaments
POPSIZE = 1000 #(250) Number of active individuals in a generation
CXPB = 0.5 #(0.5) Probability with which two individuals are crossed
MUTPB = 0.4 #(0.2) Probability for mutating an individual

scoreGoal = 0.02 #(0.02) early stop condition
###############################################################################
dataset = 'dekois'

prefix = os.getcwd() + '/' + dataset + '/'

def opt(target_id):
    stopEarly = False
    record = []
    t0 = time()
    random.seed(42)
    np.random.seed(42)
    picklePrintName = prefix + target_id + '_unsplitDataFrame.pkl'
    pickleDistName = prefix + target_id + '_distances.pkl'
    #print('Target: {}'.format(target_id))
    #print('Reading data...')
    distanceMatrix = pd.read_pickle(pickleDistName)
    #Get features
    features = pd.read_pickle(picklePrintName)
    data = ukyScore.data_set(distanceMatrix, features)
    bestSplit = (np.zeros(data.size), 2.0)
    with warnings.catch_warnings():
    #Suppress warning from rewriting optimizer
        warnings.simplefilter("ignore")
        creator.create("FitnessMin{}".format(target_id), base.Fitness, weights=(-1.0,))
        creator.create("Individual{}".format(target_id), np.ndarray, fitness=eval("creator.FitnessMin{}".format(target_id)))
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", np.random.choice,
                         2, p=[1-targetRatio , targetRatio])
        toolbox.register("individual{}".format(target_id), tools.initRepeat, eval("creator.Individual{}".format(target_id)),
                         toolbox.attr_bool, data.size)
        toolbox.register("population{}".format(target_id), tools.initRepeat, list, eval('toolbox.individual{}'.format(target_id)))
        toolbox.register("evaluate{}".format(target_id), data.evaluate)
        toolbox.register("mate{}".format(target_id), tools.cxOnePoint)
        toolbox.register("mutate{}".format(target_id), tools.mutFlipBit, indpb=INDPB)
        toolbox.register("select{}".format(target_id), tools.selTournament, tournsize=TOURNSIZE)

    #print('Initializing population...')
    pop = eval('toolbox.population{}(n=POPSIZE)'.format(target_id))

    fitnesses = list(map(eval('toolbox.evaluate{}'.format(target_id)), pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    gen = 0
    while (gen < NUMGENS) and (stopEarly == False):
        if (gen % PRINTFREQ) == 0:
            '''print statement'''
            Pop = pd.DataFrame(pop)
            validPop = Pop[Pop.apply(
                    lambda x: data.validSplit(x,
                                              targetRatio,
                                              ratioTol,
                                              balanceTol), axis=1)]
            validPop = validPop.drop_duplicates()
            numUnique = len(validPop)
            if numUnique == 0:
                meanScore = np.nan
                minScore = np.nan
                #var = np.nan
            else:
                scores = validPop.apply(lambda x: data.computeScore(x), axis=1)
                meanScore = np.mean(scores.values)
                minScore = np.min(scores.values)
                if numUnique == 1:
                    pass
                    #var = 0.0
                else:
                    pass
                    #var = validPop.var().mean()
                if minScore < bestSplit[1]:
                    minSplit = validPop.T[scores.idxmin()].values
                    if len(np.shape(minSplit)) > 1:
                        minSplit = minSplit[0,:]
                    bestSplit = (minSplit, minScore)
            print('Target: {}'.format(target_id)
                    #+ '-- Generation {}'.format(gen)
                    + ' -- Time (min): {}'.format( np.round((time()-t0)/60, 1))
                    + ' -- Min: {}'.format(np.round(minScore,4))
                    + ' -- Mean: {}'.format(np.round(meanScore,4))
                    #+ ' -- Unique Valid: {}/{}'.format(numUnique, POPSIZE)
                    #+ ' -- Var: {}'.format(np.round(var, 4))
                    )
            record.append((time()-t0, np.round(minScore,4)))
            if minScore < scoreGoal:
                stopEarly = True
                print('Goal met: {} (stopping early)'.format(target_id))
        # A new generation
        gen += 1
        # Select the next generation individuals
        offspring = eval('toolbox.select{}(pop, len(pop))'.format(target_id))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                eval('toolbox.mate{}(child1, child2)'.format(target_id))
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < MUTPB:
                eval('toolbox.mutate{}(mutant)'.format(target_id))
                del mutant.fitness.values
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(eval('toolbox.evaluate{}'.format(target_id)), invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = offspring

    print('Saving best split and record: {}'.format(target_id))
    pd.DataFrame(record).to_pickle(prefix + target_id + '_optRecord.pkl')
    Pop = pd.DataFrame(pop)
    validPop = Pop[Pop.apply(
                    lambda x: data.validSplit(x,
                                              targetRatio,
                                              ratioTol,
                                              balanceTol), axis=1)]
    validPop = validPop.drop_duplicates()
    numUnique = len(validPop)
    if numUnique == 0:
        #meanScore = np.nan
        minScore = np.nan
        #var = np.nan
    else:
        scores = validPop.apply(lambda x: data.computeScore(x), axis=1)
        #meanScore = np.mean(scores.values)
        minScore = np.min(scores.values)
        #var = validPop.var().mean()
        if minScore < bestSplit[1]:
            minSplit = validPop.T[scores.idxmin()].values
            if len(np.shape(minSplit)) > 1:
                minSplit = minSplit[0,:]
            bestSplit = (minSplit, minScore)
    pd.DataFrame(np.append(bestSplit[0], bestSplit[1])).T.to_pickle(prefix + target_id + '_optSplit.pkl')

def main():
    print('Optimizer parameters: INDPB={}, TOURNSIZE= {}, POPSIZE={}, CXPB={}, MUTPB={}'.format(
            INDPB, TOURNSIZE, POPSIZE, CXPB, MUTPB))
    files = glob(prefix + '*_unsplitDataFrame.pkl')
    targets = sorted(list(set([f.split('_')[0].split('/')[-1] for f in files])))
    #targets = targets[:2]
    with Pool() as p:
        p.map(opt, targets)


if __name__ == '__main__':
    main()
