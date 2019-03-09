importlib.reload(ukyScore)
import pandas as pd

picklePrintName = '/home/brian/Desktop/MLmodeling/DataSets/dekois/11betaHSD1_unsplitDataFrame.pkl'
pickleDistName = '/home/brian/Desktop/MLmodeling/DataSets/dekois/11betaHSD1_distances.pkl'

distanceMatrix = pd.read_pickle(pickleDistName)
features = pd.read_pickle(picklePrintName)
data = ukyScore.data_set(distanceMatrix, features)

opt = ukyScore.geneticOptimizer(data)
