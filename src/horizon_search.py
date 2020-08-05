from src.train import *
from src.data.preprocess import preprocess
from src.visualization.visualize import plot_horizon_search
import pandas as pd
import os
import yaml
import datetime

def horizon_search():
    '''
    Experiment to demonstrate the effect of the predictive horizon on model performance on the test set.
    Trains models at different values of predictive horizon (N) in weeks.
    Set experiment options in config.yml.
    '''

    # Load relevant values from config
    input_stream = open(os.getcwd() + "/config.yml", 'r')
    cfg = yaml.full_load(input_stream)
    N_MIN = cfg['HORIZON_SEARCH']['N_MIN']
    N_MAX = cfg['HORIZON_SEARCH']['N_MAX']
    N_INTERVAL = cfg['HORIZON_SEARCH']['N_INTERVAL']

    test_metrics_df = pd.DataFrame()
    for n in range(N_MIN, N_MAX + N_INTERVAL, N_INTERVAL):
        print('** n = ', n, ' of ', N_MAX)

        # Preprocess data. Avoid recomputing ground truths and classifying features after first iteration.
        if n == N_MIN:
            preprocess(n_weeks=n, calculate_gt=True, classify_cat_feats=True, load_ct=False)
        else:
            if cfg['TRAIN']['MODEL_DEF'] == 'hifis_rnn_mlp':
                preprocess(n_weeks=n, calculate_gt=True, classify_cat_feats=False, load_ct=True)
            else:
                preprocess(n_weeks=n, calculate_gt=False, classify_cat_feats=False, load_ct=True)

        # Conduct cross validation at this prediction horizon
        callbacks = define_callbacks(cfg)
        if cfg['TRAIN']['MODEL_DEF'] == 'hifis_rnn_mlp':
            results_df = nested_cross_validation(cfg, callbacks, None)  # If time series data, do nested CV
        else:
            results_df = kfold_cross_validation(cfg, callbacks, None)  # If not time series data, do k-fold CV
        results_df = results_df[0:-2]   # Remove rows for mean and std dev
        results_df.drop('Fold', axis=1, inplace=True)   # Remove fold column
        results_df.insert(0, 'n', n)  # Add prediction horizon to test results
        test_metrics_df = test_metrics_df.append(results_df)  # Append results from this value of n

    # Save results
    test_metrics_df.to_csv(cfg['PATHS']['HORIZON_SEARCH'] + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv',
                           sep=',', header=True, index=False)

    # Plot results
    plot_horizon_search(test_metrics_df, cfg['PATHS']['IMAGES'])

if __name__ == '__main__':
    results = horizon_search()
