#!/usr/bin/env python
"""
    Experiment 8: classical detector baselines (Isolation Forest, LOF,
    One-Class SVM) on the identical chronological splits and windows.

    Purpose: a reference for the ranking power of the autoencoders
    (ROC-AUC, PR-AUC, oracle F1*).
    Fitted on the same normal-only training windows (flattened), scored on
    the same test windows; three seeds where the method is stochastic.

    Usage:
        python experiments/exp_8_classical_baselines.py --dataset skab --data_path ./data/SKAB/
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments import pipeline as pl


def optimal_f1(y, scores):
    p, r, _ = precision_recall_curve(y, scores)
    f1 = 2 * p * r / (p + r + 1e-8)
    return float(np.max(f1[:-1]))


def run(args):
    split = pl.prepare_split(args.dataset, args.data_path, 0.2, 0.2)
    Xtr = split['X_train_normal'].reshape(len(split['X_train_normal']), -1)
    # LOF/OC-SVM are O(n^2)-ish in fit; subsample the (shuffled) normal
    # training windows to 5,000 for them on large datasets, disclosed in the
    # paper text. IsolationForest uses the full set.
    rng = np.random.RandomState(0)
    Xtr_sub = Xtr[rng.permutation(len(Xtr))[:5000]] if len(Xtr) > 5000 else Xtr
    Xte = split['X_test'].reshape(len(split['X_test']), -1)
    y = split['y_test']

    rows = []
    for seed in [42, 43, 44]:
        models = {
            'IsolationForest': IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1),
            'LOF': LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1),
            'OneClassSVM': OneClassSVM(nu=0.05, kernel='rbf', gamma='scale'),
        }
        for name, m in models.items():
            if name != 'IsolationForest' and seed != 42:
                continue  # LOF/OCSVM deterministic; run once
            m.fit(Xtr if name == 'IsolationForest' else Xtr_sub)
            s = -m.score_samples(Xte) if hasattr(m, 'score_samples') else -m.decision_function(Xte)
            rows.append({'dataset': args.dataset, 'method': name, 'seed': seed,
                         'roc_auc': roc_auc_score(y, s),
                         'pr_auc': average_precision_score(y, s),
                         'f1_optimal': optimal_f1(y, s)})
            print(rows[-1])

    df = pd.DataFrame(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir,
                       f"exp8_classical_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(out, index=False)
    print('saved', out)
    print(df.groupby('method')[['roc_auc', 'pr_auc', 'f1_optimal']].agg(['mean', 'std']).round(3))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['skab', 'swat'])
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--output_dir', default='./results/classical_baselines')
    args = ap.parse_args()
    run(args)
