
import random
import numpy as np
import pandas as pd
from numba import jit

#from deap import algorithms
from deap import base
from deap import creator
from deap import tools

###############################################################################
#Define evaluator

n = None # default n=None. For approx. by Atomwise is 50
#biasTolerance = 1e-3 #(1e-3) bias bound which causes early termination
split_ratio = 0.8 #(0.8) percent of data used for training
#splitSizeTolerance = 0.05 #(0.05) split size error which causes early termination 

dataSize = 2377
optimalTrainingSize = dataSize*split_ratio



#Get distance matrix
distanceMatrix = pd.read_csv('distanceMatrix.csv', index_col=0).values
#Get labels
Labels = pd.read_csv('labels.csv', index_col=0).values.flatten()

#dataSplit = np.random.randint(2,size=(dataSize,))

def AVE_bias(dataSplit):
    minPosPosDist = np.amin(distanceMatrix[(dataSplit==0) & (Labels==1),:]\
                               [:,(dataSplit==1) & (Labels==1)],
                               axis=1)
    minPosNegDist = np.amin(distanceMatrix[(dataSplit==0) & (Labels==1),:]\
                               [:,(dataSplit==1) & (Labels==0)],
                               axis=1)
    validationPositives = np.shape(minPosPosDist)[0] \
                          + np.shape(minPosNegDist)[0]
    minNegPosDist = np.amin(distanceMatrix[(dataSplit==0) & (Labels==0),:]\
                               [:,(dataSplit==1) & (Labels==1)],
                               axis=1)
    minNegNegDist = np.amin(distanceMatrix[(dataSplit==0) & (Labels==0),:]\
                               [:,(dataSplit==1) & (Labels==0)],
                               axis=1)
    validationNegatives = np.shape(minNegPosDist)[0] \
                          + np.shape(minNegNegDist)[0]
    minDistances = [minPosPosDist,
                    minPosNegDist,
                    minNegPosDist,
                    minNegNegDist]
    
    if isinstance(n, int) and n>0:
        minDistances = [np.floor(n*D) for D in minDistances]
        minDistances = [np.sum(D)/n for D in minDistances]
    else:
        minDistances = [np.sum(D) for D in minDistances]
    
    biasAVE = (minDistances[1] - minDistances[0])/validationPositives \
              + (minDistances[2] - minDistances[3])/validationNegatives
    trainingSizeAcc = np.sum(dataSplit) / optimalTrainingSize
    return np.abs(1-trainingSizeAcc) + np.abs(biasAVE) ,

###############################################################################

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_bool, dataSize)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", AVE_bias)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.075)
toolbox.register("select", tools.selTournament, tournsize=3)

###############################################################################

def main():
    random.seed(42)
    pop = toolbox.population(n=200)
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2
    # Extracting all the fitnesses of 
    fits = [ind.fitness.values for ind in pop]
    
    # Variable keeping track of the number of generations
    g = 0
    #earlyStop = False
    # Begin the evolution
    #while (g < 1000) and (earlyStop == False):
    while g < 10:
        # A new generation
        g += 1
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
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values for ind in pop]
        #if (float(fit[0]) < splitSizeTolerance) & (fit[1] < biasTolerance):
            #earlyStop = True
        if g%1==0:
            print("-- Generation {} -- Split accuracy + AVE bias : {}".format(
                    g, np.round(min(fits),4)))
            
if __name__ == '__main__':
    main()