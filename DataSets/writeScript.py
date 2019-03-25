import os
from glob import glob


dataset = 'DUDE'
prefix = os.getcwd() + '/DataSets/' + dataset + '/'
files = glob(os.getcwd() + '/DataSets/' + dataset + '/*')
targets = []
sizes = []
for file in files:
    target_id = file.split('/')[-1].split('_')[0]
    targets.append(target_id)
targets = list(set(targets))
for target_id in targets:
    if dataset == 'dekois':
        activeFile = prefix + 'ligands/' + target_id + '.sdf.gz'
        decoyFile = prefix + 'decoys/' + target_id + '_Celling-v1.12_decoyset.sdf.gz'
    elif dataset == 'DUDE':
        activeFile = prefix + target_id + '/actives_final.sdf.gz'
        decoyFile = prefix + target_id + '/decoys_final.sdf.gz'
    elif dataset == 'MUV':
        activeFile = prefix + target_id + '_actives.sdf.gz'
        decoyFile = prefix + target_id + '_decoys.sdf.gz'
    try:
        size = os.path.getsize(activeFile)*os.path.getsize(decoyFile)
        sizes.append((size, target_id))
    except FileNotFoundError:
        pass
sizes.sort()

calls = ['# !/bin/bash\ntsp -S 16\n']
for pair in sizes:
    target_id = pair[1]
    call = f'tsp python evaluation.py {dataset} {target_id} \n'
    calls.append(call)

f = open(os.getcwd() + '/DataSets/runScript', 'w+')
f.writelines(calls)
f.close()
