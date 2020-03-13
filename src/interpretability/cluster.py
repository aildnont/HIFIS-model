import numpy as np
import yaml
import os
import pandas as pd
from datetime import datetime
from kmodes.kprototypes import KPrototypes

def cluster_clients():

    # Load project config data
    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

    # Load training and validation sets
    try:
        df = pd.read_csv(cfg['PATHS']['TRAIN_SET'])
    except FileNotFoundError:
        print("No file found at " + cfg['PATHS']['TRAIN_SET'] + ". Train a model before running this script.")
        return
    df.drop(['ClientID', 'GroundTruth'], axis=1)

    # Load feature info
    try:
        cfg_feats = yaml.full_load(open(os.getcwd() + cfg['PATHS']['DATA_INFO'], 'r'))
    except FileNotFoundError:
        print("No file found at " + cfg['PATHS']['DATA_INFO'] + ". Run preprocessing script before running this script.")
        return
        
    # Get list of categorical feature indices
    noncat_feat_idxs = [df.columns.get_loc(c) for c in cfg_feats['NON_CAT_FEATURES'] if c in df]
    cat_feat_idxs = [i for i in range(len(df.columns)) if i not in noncat_feat_idxs]

    # Run k-prototypes algorithm on all clients
    k_prototypes = KPrototypes(n_clusters=10, verbose=2, n_init=1, n_jobs=1)
    clusters = k_prototypes.fit_predict(np.array(df), categorical=cat_feat_idxs)
    print("K-prototypes cost =", k_prototypes.cost_)
    print("K-prototypes number of iterations =", k_prototypes.n_iter_)

    # Get centroids of clusters
    cluster_centroids = np.concatenate((k_prototypes.cluster_centroids_[0], k_prototypes.cluster_centroids_[1]), axis=1)
    clusters_df = pd.DataFrame(cluster_centroids, columns=list(df.columns))
    clusters_df.to_csv('results/experiments/exp_clusters_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv')

    import matplotlib.pyplot as plt
    epoch_costs = k_prototypes.epoch_costs_
    plt.plot(np.arange(0, len(epoch_costs)), epoch_costs)
    return

if __name__ == '__main__':
    cluster_clients()


