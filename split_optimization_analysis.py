from rdkit import Chem
from rdkit.Chem import AllChem
import gzip
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import pairwise_distances, jaccard_similarity_score, precision_recall_curve, auc, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from ukySplit import ukyDataSet

holdout_ratio = 1/8


def finger(mol):
    fprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    return list(fprint)


def make_prints(s):
    inf = gzip.open(s)
    gzsuppl = Chem.ForwardSDMolSupplier(inf)
    mols = [x for x in gzsuppl if x is not None]
    prints = [finger(mol) for mol in mols]
    prints = pd.DataFrame(prints).dropna().drop_duplicates().values
    return prints


def nn_similarity(probs, nn_preds):
    similarities = []
    for t in np.linspace(0, 1, num=100):
        preds = (probs > t).astype(bool)
        similarities.append(jaccard_similarity_score(preds, nn_preds))
    return np.max(similarities)


def PR_AUC(probs, labels, weights=None):
    if weights is None:
        precision, recall, _ = precision_recall_curve(labels, probs)
    else:
        precision, recall, _ = precision_recall_curve(labels, probs, sample_weight=weights)
    return auc(recall, precision)


def model_performance_scores(training_features, training_labels,
                             validation_features, validation_labels,
                             test_features, test_labels,
                             weights):
    # random forest metrics
    rf = RandomForestClassifier(n_estimators=100).fit(training_features, training_labels)
    validation_probs = rf.predict_proba(validation_features)[:, 1]
    test_probs = rf.predict_proba(test_features)[:, 1]
    validation_PR_AUC = PR_AUC(validation_probs, validation_labels)
    test_PR_AUC = PR_AUC(test_probs, test_labels)
    weighted_PR_AUC = PR_AUC(validation_probs, validation_labels, weights)

    # nearest neighbor metrics
    nn = KNeighborsClassifier(n_neighbors=1).fit(training_features, training_labels)
    nn_validation_predictions = nn.predict(validation_features).astype(bool)
    nn_test_predictions = nn.predict(test_features).astype(bool)
    nn_validation_F1 = f1_score(validation_labels, nn_validation_predictions)
    nn_test_F1 = f1_score(test_labels, nn_test_predictions)

    # rf-nn similarity
    validation_nn_similarity = nn_similarity(validation_probs, nn_validation_predictions)
    test_nn_similarity = nn_similarity(test_probs, nn_test_predictions)

    return np.array([validation_PR_AUC, test_PR_AUC, weighted_PR_AUC,
                     validation_nn_similarity, test_nn_similarity,
                     nn_validation_F1, nn_test_F1])


def main(decoys_file, actives_file, output_file):
    active_prints = make_prints(actives_file)
    decoy_prints = make_prints(decoys_file)
    fingerprints = np.vstack([active_prints, decoy_prints])
    labels = np.vstack([np.ones((len(active_prints), 1)), np.zeros((len(decoy_prints), 1))]).flatten().astype(int)

    # A full splitting of the data is into test, holdout, training, and validation sets.
    # First a random split into test and temp_training, then selecting the closest temp training data to the test data
    # as holdout data. The remaining data is split into training and validation using either a random split or optimization.

    outer_skf = StratifiedKFold(n_splits=5, shuffle=True)
    outer_random_splits = [(train, test) for train, test in outer_skf.split(fingerprints, labels)]
    results_list = []
    for temp_training_indices, test_indices in outer_random_splits:
        temp_training_fingerprints = fingerprints[temp_training_indices]
        test_fingerprints = fingerprints[test_indices]
        test_labels = labels[test_indices]
        temp_training_test_distances = np.min(pairwise_distances(temp_training_fingerprints.astype(bool),
                                                                 test_fingerprints.astype(bool),
                                                                 metric='jaccard'),
                                              axis=1
                                              )
        holdout_indices = np.sort(np.argsort(temp_training_test_distances)[:int(holdout_ratio * len(temp_training_indices))])
        temp_training_indices = temp_training_indices[~np.isin(temp_training_indices, holdout_indices)]

        temp_training_fingerprints = fingerprints[temp_training_indices]
        temp_training_labels = labels[temp_training_indices]

        AVE_dataset = ukyDataSet(temp_training_fingerprints, temp_training_labels)
        VE_dataset = ukyDataSet(temp_training_fingerprints, temp_training_labels, AVE=False)

        # AVE bias optimization first.
        AVE_training_indices, AVE_validation_indices = AVE_dataset.geneticOptimizer(1)
        AVE_split = np.isin(np.arange(AVE_dataset.size), AVE_training_indices)
        AVE_training_fingerprints = temp_training_fingerprints[AVE_training_indices]
        AVE_training_labels = temp_training_labels[AVE_training_indices]
        AVE_validation_fingerprints = temp_training_fingerprints[AVE_validation_indices]
        AVE_validation_labels = temp_training_labels[AVE_validation_indices]
        AVE_validation_weights = AVE_dataset.get_validation_weights(AVE_training_indices, AVE_validation_indices)
        AVE_perf = model_performance_scores(AVE_training_fingerprints, AVE_training_labels,
                                            AVE_validation_fingerprints, AVE_validation_labels,
                                            test_fingerprints, test_labels,
                                            AVE_validation_weights)
        AVE_perf = np.append(AVE_perf, [AVE_dataset.computeScore(AVE_split)[0], VE_dataset.computeScore(AVE_split)[0]])

        # VE score optimization next.
        VE_training_indices, VE_validation_indices = VE_dataset.geneticOptimizer(1)
        VE_split = np.isin(np.arange(VE_dataset.size), VE_training_indices)
        VE_training_fingerprints = temp_training_fingerprints[VE_training_indices]
        VE_training_labels = temp_training_labels[VE_training_indices]
        VE_validation_fingerprints = temp_training_fingerprints[VE_validation_indices]
        VE_validation_labels = temp_training_labels[VE_validation_indices]
        VE_validation_weights = VE_dataset.get_validation_weights(VE_training_indices, VE_validation_indices)
        VE_perf = model_performance_scores(VE_training_fingerprints, VE_training_labels,
                                           VE_validation_fingerprints, VE_validation_labels,
                                           test_fingerprints, test_labels,
                                           VE_validation_weights)
        VE_perf = np.append(VE_perf, [AVE_dataset.computeScore(VE_split)[0], VE_dataset.computeScore(VE_split)[0]])

        # Random splits last.
        inner_skf = StratifiedKFold(n_splits=5, shuffle=True)
        inner_random_splits = [(train, test) for train, test in inner_skf.split(temp_training_fingerprints,
                                                                                temp_training_labels)]
        collect_scores = []
        for random_training_indices, random_validation_indices in inner_random_splits:
            random_split = np.isin(np.arange(AVE_dataset.size), random_training_indices)
            random_training_fingerprints = temp_training_fingerprints[random_training_indices]
            random_training_labels = temp_training_labels[random_training_indices]
            random_validation_fingerprints = temp_training_fingerprints[random_validation_indices]
            random_validation_labels = temp_training_labels[random_validation_indices]
            random_validation_weights = AVE_dataset.get_validation_weights(random_training_indices, random_validation_indices)
            random_perf = model_performance_scores(random_training_fingerprints, random_training_labels,
                                                   random_validation_fingerprints, random_validation_labels,
                                                   test_fingerprints, test_labels,
                                                   random_validation_weights).tolist()
            random_perf = np.append(random_perf,
                                    [AVE_dataset.computeScore(random_split)[0],
                                     VE_dataset.computeScore(random_split)[0]]
                                    )
            collect_scores.append(random_perf)
        random_perf = np.mean(np.array(collect_scores), axis=0)
        results_list.append(np.hstack([AVE_perf, VE_perf, random_perf]))
    perf_names = ['validation_PR_AUC', 'test_PR_AUC', 'weighted_PR_AUC',
                  'validation_nn_similarity', 'test_nn_similarity',
                  'nn_validation_F1', 'nn_test_F1',
                  'AVE_score', 'VE_score']
    column_names = [t + s for t in ['AVE_', 'VE_', 'random_'] for s in perf_names]
    target_results = pd.DataFrame(results_list, columns=column_names)
    target_results.mean().to_csv(output_file, header=True)


if __name__ == '__main__':
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Specify decoys file, actives file, and output file.")
