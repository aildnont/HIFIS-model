import numpy as np
import yaml
import os
import dill
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from kmodes.kprototypes import KPrototypes
from tensorflow.keras.models import load_model
from sklearn.externals.joblib import load
from src.interpretability.lime_explain import predict_and_explain

def cluster_clients():

    # Load project config data
    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

    # Load training and validation sets
    try:
        df = pd.read_csv(cfg['PATHS']['TRAIN_SET'])
    except FileNotFoundError:
        print("No file found at " + cfg['PATHS']['TRAIN_SET'] + ". Train a model before running this script.")
        return
    df.drop(['ClientID', 'GroundTruth'], axis=1, inplace=True)

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
    k_prototypes = KPrototypes(n_clusters=cfg['K-PROTOTYPES']['K'], verbose=2, n_init=cfg['K-PROTOTYPES']['N_RUNS'],
                               n_jobs=cfg['K-PROTOTYPES']['N_JOBS'])
    clusters = k_prototypes.fit_predict(np.array(df), categorical=cat_feat_idxs)
    print("K-prototypes cost =", k_prototypes.cost_)
    print("K-prototypes number of iterations =", k_prototypes.n_iter_)

    # Get centroids of clusters
    cluster_centroids = np.concatenate((k_prototypes.cluster_centroids_[0], k_prototypes.cluster_centroids_[1]), axis=1)
    feat_names = list(df.columns)

    # Predict and explain the cluster centroids
    NUM_SAMPLES = cfg['LIME']['NUM_SAMPLES']
    NUM_FEATURES = cfg['LIME']['NUM_FEATURES']
    scaler_ct = load(cfg['PATHS']['SCALER_COL_TRANSFORMER'])
    ohe_ct_sv = load(cfg['PATHS']['OHE_COL_TRANSFORMER_SV'])
    explainer = dill.load(open(cfg['PATHS']['LIME_EXPLAINER'], 'rb'))
    model = load_model(cfg['PATHS']['MODEL_TO_LOAD'])
    exp_rows = []
    print('Creating explanations for cluster centroids.')
    for i in tqdm(range(cluster_centroids.shape[0])):
        row = []
        explanation = predict_and_explain(cluster_centroids[i], model, explainer, ohe_ct_sv, scaler_ct, NUM_FEATURES, NUM_SAMPLES)
        exp_tuples = explanation.as_list()
        for exp_tuple in exp_tuples:
            row.extend(list(exp_tuple))
        if len(exp_tuples) < NUM_FEATURES:
            row.extend([''] * (2 * (NUM_FEATURES - len(exp_tuples))))  # Fill with empty space if explanation too small
        exp_rows.append(row)
    exp_col_names = []
    for i in range(NUM_FEATURES):
        exp_col_names.extend(['Explanation ' + str(i + 1), 'Weight ' + str(i + 1)])
    exp_df = pd.DataFrame(exp_rows, columns=exp_col_names)

    clusters_df = pd.DataFrame(cluster_centroids, columns=feat_names)
    clusters_df = pd.concat([clusters_df, exp_df], axis=1, sort=False)
    clusters_df.to_csv(cfg['PATHS']['K-PROTOTYPES'] + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv',
                       index_label=False, index=False)
    return

if __name__ == '__main__':
    cluster_clients()


