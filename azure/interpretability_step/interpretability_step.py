import os
import yaml
import argparse
from src.interpretability.lime_explain import setup_lime, submodular_pick

parser = argparse.ArgumentParser()
parser.add_argument('--preprocessedoutputdir', type=str, help="intermediate preprocessed pipeline data directory")
parser.add_argument('--trainoutputdir', type=str, help="intermediate training pipeline data directory")
parser.add_argument('--interpretabilityoutputdir', type=str, help="intermediate interpretability pipeline data directory")
args = parser.parse_args()

# Update paths of input data in config to represent paths on blob.
cfg = yaml.full_load(open("./config.yml", 'r'))  # Load config data
cfg['PATHS']['TRAIN_SET'] = args.preprocessedoutputdir + '/' + cfg['PATHS']['TRAIN_SET'].split('/')[-1]
cfg['PATHS']['TEST_SET'] = args.preprocessedoutputdir + '/' + cfg['PATHS']['TEST_SET'].split('/')[-1]
cfg['PATHS']['DATA_INFO'] = args.preprocessedoutputdir + '/' + cfg['PATHS']['DATA_INFO'].split('/')[-1]
cfg['PATHS']['OHE_COL_TRANSFORMER_SV'] = args.preprocessedoutputdir + '/' + cfg['PATHS']['OHE_COL_TRANSFORMER_SV'].split('/')[-1]
cfg['PATHS']['SCALER_COL_TRANSFORMER'] = args.trainoutputdir + '/' + cfg['PATHS']['SCALER_COL_TRANSFORMER'].split('/')[-1]
cfg['PATHS']['IMAGES'] = args.interpretabilityoutputdir + '/' + cfg['PATHS']['IMAGES'].split('/')[-1]
cfg['PATHS']['MODEL_TO_LOAD'] = args.trainoutputdir + '/' + cfg['PATHS']['MODEL_TO_LOAD'].split('/')[-1]
cfg['PATHS']['LIME_EXPLAINER'] = args.interpretabilityoutputdir + '/' + cfg['PATHS']['LIME_EXPLAINER'].split('/')[-1]
cfg['PATHS']['LIME_SUBMODULAR_PICK'] = args.interpretabilityoutputdir + '/' + cfg['PATHS']['LIME_SUBMODULAR_PICK'].split('/')[-1]

# Create path for interpretability step output data if it doesn't already exist on the blob
if not os.path.exists(args.interpretabilityoutputdir):
    os.makedirs(args.interpretabilityoutputdir)

# Fit LIME explainer object and run submodular pick.
lime_dict = setup_lime(cfg=cfg)
submodular_pick(lime_dict, file_path=cfg['PATHS']['IMAGES'] + '/submodular_pick.png')