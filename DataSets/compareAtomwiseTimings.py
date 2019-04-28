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


terminatedTargets = []
notTerminatedTargets = []
for target_id in targets:
    optRecord = pd.read_pickle(os.getcwd() + '/DataSets/dekois/' + target_id + '_optRecord.pkl')
    finalScore = list(optRecord.tail(1)[['time', 'score']].values[0])
    if finalScore[-1] < 0.02:
        terminatedTargets.append([target_id] + finalScore)

    else:
        notTerminatedTargets.append([target_id] + finalScore)

collectNonTerminating = []
for data in notTerminatedTargets:
    try:
        target_id = data[0]
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
        stopTime = data[1]
        atomwiseStop = atomwise[atomwise[0] < stopTime].tail(1).values[0]
        data.extend(atomwiseStop)
        atomwiseFull = atomwise.tail(1).values[0]
        data.extend(atomwiseFull)
        collectNonTerminating.append(data)
    except:
        pass

collectNonTerminating = pd.DataFrame(collectNonTerminating)

plt.scatter(collectNonTerminating[2], collectNonTerminating[4], label='Atomwise Compare')
plt.scatter(collectNonTerminating[2], collectNonTerminating[6], label='Atomwise Terminal')
plt.plot([0, 1], [0, 1], ls='--', c='k')
plt.legend()
plt.xlabel('ukySplit Terminal AVE Bias')
plt.ylabel('Atomwise AVE Bias')

collectNonTerminating[1].mean()/(60)
collectNonTerminating[5].mean()/60
########################################################################################################################

collectTerminating = []
for data in terminatedTargets:
    try:
        target_id = data[0]
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
        stopTime = data[1]
        atomwiseStop = atomwise[atomwise[0] < stopTime].tail(1).values[0]
        data.extend(atomwiseStop)
        atomwiseFull = atomwise.tail(1).values[0]
        data.extend(atomwiseFull)
        collectTerminating.append(data)
    except:
        pass

collectTerminating = pd.DataFrame(collectTerminating)

atomwiseTerminating = collectTerminating[collectTerminating[6] < 0.02]
plt.scatter(atomwiseTerminating[1]/60, atomwiseTerminating[5]/60)
plt.plot([0, 25], [0, 25], ls='--', c='k', label='y = x')
plt.ylabel('Atomwise Stop Time (min)')
plt.xlabel('ukySplit Stop Time (min)')

atomwiseNonTerminating = collectTerminating[collectTerminating[6] >= 0.02]
plt.scatter(atomwiseNonTerminating[1]/60, atomwiseNonTerminating[4], label='Atomwise Compare')
plt.scatter(atomwiseNonTerminating[1]/60, atomwiseNonTerminating[6], label='Atomwise Terminal')
plt.plot([0, 45], [0.02, 0.02], ls='--', c='k')
#plt.ylim([0.02, 1])
plt.legend()
plt.xlabel('ukySplit Terminal AVE Bias')
plt.ylabel('Atomwise AVE Bias')



