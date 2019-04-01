import os
from glob import glob

import pandas as pd

dataset = 'dekois'

prefix = os.getcwd() + '/DataSets/' + dataset
files = glob(prefix + '/*.out')
targets = list(set([f.split('.')[0].split('/')[-1] for f in files]))

for target_id in targets:
    targetOutFile = prefix + '/' + target_id + '.out'
    f = open(targetOutFile)
    targetOutString = f.read()
    f.close()
    targetOutString = targetOutString.split('time: ')[2:-1]
    acumLog = []
    for string in targetOutString:
        s, t = string.split('minObj= ')
        acumLog.append((float(s.split(' sec')[0]), float(t.split('\n')[0])))
    pd.DataFrame(acumLog).to_pickle(prefix + '/' + target_id + '_atomwiseLog.pkl')

