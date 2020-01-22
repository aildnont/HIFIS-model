from src.train import train_model
from src.data.preprocess import preprocess
from src.visualization.visualize import plot_horizon_search
import pandas as pd
import os
import yaml

def horizon_search():
    # Load relevant values from config
    input_stream = open(os.getcwd() + "/config.yml", 'r')
    cfg = yaml.full_load(input_stream)
    N_MIN = cfg['DATA']['N_MIN']
    N_MAX = cfg['DATA']['N_MAX']
    N_INTERVAL = cfg['DATA']['N_INTERVAL']
    RUNS_PER_N = cfg['TRAIN']['RUNS_PER_N']

    test_metrics_df = pd.DataFrame()
    for n in range(N_MIN, N_MAX + N_INTERVAL, N_INTERVAL):
        print('** n = ', n, ' of ', N_MAX)
        # Preprocess data. Avoid recomputing ground truths and classifying features after first iteration.
        if n == N_MIN:
            preprocess(n_weeks=n, load_gt=False, classify_cat_feats=True)
        else:
            preprocess(n_weeks=n, load_gt=True, classify_cat_feats=False)

        # Train the model several times at this prediction horizon
        results_df = pd.DataFrame()
        for i in range(RUNS_PER_N):
            results = train_model(save_weights=False)
            results_df = results_df.append(pd.DataFrame.from_records([results]))
        results_df.insert(0, 'n', n)    # Add prediction horizon to test results
        test_metrics_df = test_metrics_df.append(results_df)    # Append results from this value of n
    #test_metrics_df.drop(test_metrics_df.columns[0],axis=1,inplace=True)    # Drop dummy first column

    # Save results
    test_metrics_df.to_csv(cfg['PATHS']['HORIZON_SEARCH'], sep=',', header=True, index=False)

    # Plot results
    plot_horizon_search(test_metrics_df, cfg['PATHS']['IMAGES'])

if __name__ == '__main__':
    results = horizon_search()
