import os
import sys
from glob import glob


dataset = 'MUV'
files = glob(os.getcwd() + '/DataSets/' + dataset + '/*_actives.sdf.gz')
calls = ['# !/bin/bash\n']
for file in files:
    target_id = file.split('/')[-1].split('_')[0]
    call = f'python singleTargetProcess.py {dataset} {target_id} & \ \n'
    calls.append(call)

f = open(os.getcwd() + '/DataSets/runScript', 'w+')
f.writelines(calls)
f.close()






