import numpy as np
import pandas as pd
from glob import glob

dataset = 'dekois'
files = glob(dataset + '/*_dataPackage.pkl')
file = files[0]
package = pd.read_pickle(file)
