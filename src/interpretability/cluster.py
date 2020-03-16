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
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from src.interpretability.lime_explain import predict_and_explain
from src.visualization.visualize import visualize_multiple_explanations

def cluster_clients(save_centroids=True, save_clusters=True, explain_centroids=True):
    '''
    Runs k-prototype clustering algorithm on preprocessed dataset
    :param save_centroids: Boolean indicating whether to save cluster centroids
    :param save_clusters: Boolean indicating whether to save client cluster assignments
    :param explain_centroids:
    :return:
    '''

    # Load project config data
    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

    # Load preprocessed client data
    try:
        df = pd.read_csv(cfg['PATHS']['PROCESSED_DATA'])
    except FileNotFoundError:
        print("No file found at " + cfg['PATHS']['PROCESSED_DATA'] + ". Run preprocessing script before running this script.")
        return
    client_ids = df.pop('ClientID').tolist()
    df.drop('GroundTruth', axis=1, inplace=True)
    X = np.array(df)

    # Load feature info
    try:
        cfg_feats = yaml.full_load(open(os.getcwd() + cfg['PATHS']['DATA_INFO'], 'r'))
    except FileNotFoundError:
        print("No file found at " + cfg['PATHS']['DATA_INFO'] + ". Run preprocessing script before running this script.")
        return
        
    # Get list of categorical feature indices
    noncat_feat_idxs = [df.columns.get_loc(c) for c in cfg_feats['NON_CAT_FEATURES'] if c in df]
    cat_feat_idxs = [i for i in range(len(df.columns)) if i not in noncat_feat_idxs]

    # Run k-prototypes algorithm on all clients and obtain cluster assignment (range [1, K]) for each client
    k_prototypes = KPrototypes(n_clusters=cfg['K-PROTOTYPES']['K'], verbose=2, n_init=cfg['K-PROTOTYPES']['N_RUNS'],
                               n_jobs=cfg['K-PROTOTYPES']['N_JOBS'], init='random')
    client_clusters = k_prototypes.fit_predict(X, categorical=cat_feat_idxs)
    client_clusters += 1    # Enforce that cluster labels are integer range of [1, K]
    clusters_df = pd.DataFrame({'ClientID': client_ids, 'Cluster Membership': client_clusters})
    clusters_df.set_index('ClientID')

    # Get centroids of clusters
    cluster_centroids = np.concatenate((k_prototypes.cluster_centroids_[0], k_prototypes.cluster_centroids_[1]), axis=1)
    feat_names = list(df.columns)
    centroids_df = pd.DataFrame(cluster_centroids, columns=feat_names)
    cluster_num_series = pd.Series(np.arange(1, cluster_centroids.shape[0] + 1))
    centroids_df.insert(0, 'Cluster', cluster_num_series)

    # Predict and explain the cluster centroids
    if explain_centroids:
        NUM_SAMPLES = cfg['LIME']['NUM_SAMPLES']
        NUM_FEATURES = cfg['LIME']['NUM_FEATURES']
        try:
            scaler_ct = load(cfg['PATHS']['SCALER_COL_TRANSFORMER'])
            ohe_ct_sv = load(cfg['PATHS']['OHE_COL_TRANSFORMER_SV'])
            explainer = dill.load(open(cfg['PATHS']['LIME_EXPLAINER'], 'rb'))
            model = load_model(cfg['PATHS']['MODEL_TO_LOAD'])
        except FileNotFoundError as not_found_err:
            print('File "' + not_found_err.filename + 
                  '" was not found. Ensure you have trained a model and run LIME before running this script.')
            return
        exp_rows = []
        explanations = []
        print('Creating explanations for cluster centroids.')
        for i in tqdm(range(cluster_centroids.shape[0])):
            row = []
            exp = predict_and_explain(cluster_centroids[i], model, explainer, ohe_ct_sv, scaler_ct, NUM_FEATURES, NUM_SAMPLES)
            explanations.append(exp)
            exp_tuples = exp.as_list()
            for exp_tuple in exp_tuples:
                row.extend(list(exp_tuple))
            if len(exp_tuples) < NUM_FEATURES:
                row.extend([''] * (2 * (NUM_FEATURES - len(exp_tuples))))  # Fill with empty space if explanation too small
            exp_rows.append(row)
        exp_col_names = []
        for i in range(NUM_FEATURES):
            exp_col_names.extend(['Explanation ' + str(i + 1), 'Weight ' + str(i + 1)])
        exp_df = pd.DataFrame(exp_rows, columns=exp_col_names)
        centroids_df = pd.concat([centroids_df, exp_df], axis=1, sort=False)    # Concatenate client features and explanations

        # Visualize clusters' LIME explanations
        visualize_multiple_explanations(explanations, 'Explanations for k-prototypes clusters',
                                        cfg['PATHS']['IMAGES'] + 'cluster_explanations')

    # Save centroid features and explanations to spreadsheet
    if save_centroids:
        centroids_df.to_csv(cfg['PATHS']['K-PROTOTYPES_CENTROIDS'] + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv',
                           index_label=False, index=False)

    if save_clusters:
        clusters_df.to_csv(cfg['PATHS']['K-PROTOTYPES_CLUSTERS'] + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv',
                           index_label=False, index=False)
    return

if __name__ == '__main__':
    cluster_clients(save_centroids=True, save_clusters=True, explain_centroids=True)


