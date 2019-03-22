import os
import sys
from glob import glob


dataset = 'dekois'
files = glob(os.getcwd() + '/DataSets/' + dataset + '/decoys/*.sdf.gz')

n = 1
calls = ['# !/bin/bash\n\n']
for file in files[16*n:16*(n+1)-1]:
    target_id = file.split('/')[-1].split('_')[0]
    call = f'python singleTargetProcess.py {dataset} {target_id} & \ \n'
    calls.append(call)
target_id = files[16*(n+1)].split('/')[-1].split('_')[0]
calls.append(f'python singleTargetProcess.py {dataset} {target_id}')

f = open(os.getcwd() + '/DataSets/runScript', 'w+')
f.writelines(calls)
f.close()






