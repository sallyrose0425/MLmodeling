import os
import sys
from glob import glob


dataset = 'dekois'
files = glob(os.getcwd() + '/DataSets/' + dataset + '/decoys/*.sdf.gz')

calls = ['# !/bin/bash\ntsp -S 16\n']
for file in files:
    target_id = file.split('/')[-1].split('_')[0]
    call = f'tsp python singleTargetProcess.py {dataset} {target_id} \n'
    calls.append(call)


f = open(os.getcwd() + '/DataSets/runScript', 'w+')
f.writelines(calls)
f.close()






