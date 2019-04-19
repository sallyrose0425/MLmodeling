import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt


prefix = os.getcwd() + '/AtomWise/dekois_out(1)/'
files = glob(prefix + '*.out')
targets = list(set([f.split('.')[0].split('/')[-1] for f in files]))


for target_id in targets:
    try:
        targetOutFile = prefix + target_id + '.out'
        f = open(targetOutFile)
        targetOutString = f.read()
        f.close()
        targetOutString = targetOutString.split('time: ')[2:-1]
        acumLog = []
        for string in targetOutString:
            s, t = string.split('minObj= ')
            acumLog.append((float(s.split(' sec')[0]), float(t.split('\n')[0])))
        atomwise = pd.DataFrame(acumLog)
        optRecord = pd.read_pickle(os.getcwd() + '/DataSets/dekois/' + target_id + '_optRecord.pkl')
        fig = plt.figure()
        plt.scatter(atomwise[0], atomwise[1], label='Atomwise')
        plt.scatter(optRecord['time'], optRecord['score'], label='New Alg.')
        plt.xlabel('Comp Time (sec)')
        plt.ylabel('AVE Bias')
        plt.legend()
        plt.title(f'{target_id} Optimization')
        plt.savefig(os.getcwd() + '/DataSets/dekois/timingComparisonFigs/' + target_id + '_comparison.png')
        plt.close(fig)
    except KeyError:
        pass
