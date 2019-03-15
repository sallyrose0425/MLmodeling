import numpy as np
import pandas as pd
from glob import glob

dataset = 'dekois'
files = glob(dataset + '/*_dataPackage.pkl')
file = files[0]
package = pd.read_pickle(file)
training = package[package['split'] == 1]
validation = package[package['split'] == 0]
trainingFeatures = training.drop(['split', 'labels', 'weights'], axis=1)
trainingLabels = package[split == 1]
validationLabels = package[split == 0]