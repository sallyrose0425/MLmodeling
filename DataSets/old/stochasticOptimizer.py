
"""
cd DataSets

from importlib import reload
import os
import ukyScore
import numpy as np
dataset = 'dekois'
target_id = 'ADRB2'
prefix = os.getcwd() + '/' + dataset + '/'
activeFile = prefix + 'ligands/' + target_id + '.sdf.gz'
decoyFile = prefix + 'decoys/' + target_id + '_Celling-v1.12_decoyset.sdf.gz'

reload(ukyScore)

data = ukyScore.data_set(activeFile, decoyFile, balanceTol=0.01)
"""

from time import time
from multiprocessing import Pool
import os
import ukyScore
import numpy as np


dataset = 'dekois'
target_id = 'ADRB2'
prefix = os.getcwd() + '/' + dataset + '/'
activeFile = prefix + 'ligands/' + target_id + '.sdf.gz'
decoyFile = prefix + 'decoys/' + target_id + '_Celling-v1.12_decoyset.sdf.gz'
data = ukyScore.data_set(activeFile, decoyFile, balanceTol=0.01)


if __name__ == '__main__':
    t0 = time()
    with Pool(14) as p:
        record = p.map(data.sample, [1000000]*14)
    print(np.min(record))
    print((time() - t0))

