from src.train import train_model
from src.data.preprocess import preprocess
import os
import yaml
import json

# Load relevant values from config
input_stream = open(os.getcwd() + "/config.yml", 'r')
cfg = yaml.full_load(input_stream)
N_MIN = cfg['DATA']['N_MIN']
N_MAX = cfg['DATA']['N_MAX']
N_INTERVAL = cfg['DATA']['N_INTERVAL']
RUNS_PER_N = cfg['TRAIN']['RUNS_PER_N']

test_metrics = {}
for n in range(N_MIN, N_MAX + N_INTERVAL, N_INTERVAL):
    print('** n = ', n, ' of ', N_MAX)
    # Preprocess data. Avoid recomputing ground truths and classifying features after first iteration.
    if n == N_MIN:
        preprocess(n_weeks=n, load_gt=False, classify_cat_feats=True)
    else:
        preprocess(n_weeks=n, load_gt=True, classify_cat_feats=False)

    # Train the model several times at this prediction horizon
    metrics_n = []
    for i in range(RUNS_PER_N):
        results = train_model(save_weights=False)
        metrics_n.append(results)
    test_metrics[n] = metrics_n

# Save results
with open(cfg['PATHS']['HORIZON_SEARCH'], 'w') as file:
    interpretability_doc = yaml.dump(test_metrics, file)
