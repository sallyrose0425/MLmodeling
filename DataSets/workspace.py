from importlib import reload
import ukyScore
import pandas as pd

reload(ukyScore)

picklePrintName = '/home/brian/Desktop/MLmodeling/DataSets/dekois/11betaHSD1_unsplitDataFrame.pkl'
pickleDistName = '/home/brian/Desktop/MLmodeling/DataSets/dekois/11betaHSD1_distances.pkl'

distanceMatrix = pd.read_pickle(pickleDistName)
features = pd.read_pickle(picklePrintName)
data = ukyScore.data_set(distanceMatrix, features)

splits = data.geneticOptimizer(1, POPSIZE=50)
