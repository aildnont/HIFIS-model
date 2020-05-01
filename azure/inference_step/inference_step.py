import os
import yaml
import argparse
import datetime
import pandas as pd
from src.predict import predict_and_explain_set

parser = argparse.ArgumentParser()
parser.add_argument('--preprocessedoutputdir', type=str, help="intermediate preprocessed pipeline data directory")
parser.add_argument('--inferencedir', type=str, help="directory containing all files necessary for inference")
args = parser.parse_args()

# Modify paths in config file based the Azure datastore paths passed as arguments.
cfg = yaml.full_load(open("./config.yml", 'r'))
cfg['PATHS']['PROCESSED_DATA'] = args.preprocessedoutputdir + '/' + cfg['PATHS']['PROCESSED_DATA'].split('/')[-1]
cfg['PATHS']['DATA_INFO'] = args.inferencedir + cfg['PATHS']['DATA_INFO'].split('/')[-1]
cfg['PATHS']['OHE_COL_TRANSFORMER_SV'] = args.inferencedir + \
                                         cfg['PATHS']['OHE_COL_TRANSFORMER_SV'].split('/')[-1]
cfg['PATHS']['SCALER_COL_TRANSFORMER'] = args.inferencedir + \
                                         cfg['PATHS']['SCALER_COL_TRANSFORMER'].split('/')[-1]
cfg['PATHS']['MODEL_TO_LOAD'] = args.inferencedir + cfg['PATHS']['MODEL_TO_LOAD'].split('/')[-1]
cfg['PATHS']['LIME_EXPLAINER'] = args.inferencedir + cfg['PATHS']['LIME_EXPLAINER'].split('/')[-1]
cfg['PATHS']['TRENDING_PREDICTIONS'] = args.inferencedir + cfg['PATHS']['TRENDING_PREDICTIONS'].split('/')[-1]

# Load preprocessed data from intermediate pipeline data
processed_df = pd.read_csv(cfg['PATHS']['PROCESSED_DATA'])
print("SHAPE", processed_df.shape)

# Preprocess raw data, run inference, and run LIME on each client in the raw data
results_df = predict_and_explain_set(cfg=cfg, data_path=None, save_results=False, give_explanations=True,
                                     include_feat_values=True, processed_df=processed_df)

results_df.insert(0, 'Timestamp', pd.to_datetime(datetime.datetime.today()))     # Add a timestamp to these results

# If previous prediction file exists, load it and append the predictions we just made.
if os.path.exists(cfg['PATHS']['TRENDING_PREDICTIONS']):
    prev_results_df = pd.read_csv(cfg['PATHS']['TRENDING_PREDICTIONS'])
    results_df = pd.concat([prev_results_df, results_df], axis=0, sort=False)

# Save the updated trend analysis prediction spreadsheet
results_df.to_csv(cfg['PATHS']['TRENDING_PREDICTIONS'], columns=list(results_df.columns), index_label=False, index=False)