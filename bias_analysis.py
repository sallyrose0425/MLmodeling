#import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.stats as st

#import dataBias

scale = StandardScaler()


###############################################################################
def main(s):
    dataset = s
    if dataset not in ['dekois', 'DUDE', 'MUV']:
        print('\n Valid data sets are dekois, DUDE, and MUV')
        return
    files = glob.glob(dataset + '/*_samples.pkl')
    targets = sorted(list(set([(f.split('_')[0]).split('/')[1] for f in files])))
    acumExp = []
    acumMean = []
    acumStd = []
    for target_id in targets:
        exp = pd.read_pickle(dataset + '/' + target_id + '_samples.pkl')
        mean = exp['bias'].mean()
        std = exp['bias'].std()
        #exp['bias'] = exp['bias'].apply(lambda x: (x-mean)/std)
        acumExp.append(exp)
        acumMean.append(mean)
        acumStd.append(std)

    #Pearson Correlation Coeff. between split size and score
    print('\n Pearson cor. coef. for score / split ratio: {}'.format(
        pd.concat(acumExp).corr()['bias'].values[1]))

    # Mean stats
    tmp = pd.DataFrame(np.array(acumMean))
    print('\n Score means (over targets): \n{}\n'.format(
        tmp.describe()[0].to_string()))
    tmp = pd.DataFrame(np.array(acumStd))
    print('\n Score std (over targets): \n{}'.format(
        tmp.describe()[0].to_string()))

    # plotting standardized aggregate scores
    acumStandardExp = [pd.DataFrame(scale.fit_transform(x)) for x in acumExp]
    combSample = pd.concat(acumStandardExp)
    x = np.linspace(-4, 4, 100)
    plt.plot(x, st.norm.pdf(x),'r-', lw=2, label='norm pdf')
    combSample[0].hist(density=True, bins=100, range=(-4,4))
    plt.title('\n {} aggregate standardized scores'.format(dataset))
    plt.show()

###############################################################################

if __name__ == '__main__':
    if len(sys.argv)>1:
        main(sys.argv[1])
    else:
        print('No data set specified...')
