import os
import yaml
import argparse
from src.data.preprocess import preprocess

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, help="Raw HIFIS client data directory")
parser.add_argument('--preprocesseddir', type=str, help="preprocessed output")
args = parser.parse_args()

# Modify paths in config file based the Azure datastore paths passed as arguments.
cfg = yaml.full_load(open(os.getcwd() + "./config.yml", 'r'))  # Load config data
cfg['PATHS']['RAW_DATA'] = args.datadir + cfg['PATHS']['RAW_DATA'][cfg['PATHS']['RAW_DATA'].index('/')+1:]
cfg['PATHS']['RAW_SPDAT_DATA'] = args.datadir + \
                                 cfg['PATHS']['RAW_SPDAT_DATA'][cfg['PATHS']['RAW_SPDAT_DATA'].index('/')+1:]
cfg['PATHS']['PROCESSED_DATA'] = args.preprocesseddir + '/' + cfg['PATHS']['PROCESSED_DATA'].split('/')[-1]
cfg['PATHS']['PROCESSED_OHE_DATA'] = args.preprocesseddir + '/' + cfg['PATHS']['PROCESSED_OHE_DATA'].split('/')[-1]
cfg['PATHS']['TRAIN_SET'] = args.preprocesseddir + '/' + cfg['PATHS']['TRAIN_SET'].split('/')[-1]
cfg['PATHS']['TEST_SET'] = args.preprocesseddir + '/' + cfg['PATHS']['TEST_SET'].split('/')[-1]
cfg['PATHS']['GROUND_TRUTH'] = args.preprocesseddir + '/' + cfg['PATHS']['GROUND_TRUTH'].split('/')[-1]
cfg['PATHS']['DATA_INFO'] = args.datadir + cfg['PATHS']['DATA_INFO'][cfg['PATHS']['DATA_INFO'].index('/'):]
cfg['PATHS']['ORDINAL_COL_TRANSFORMER'] = args.datadir + \
                                          cfg['PATHS']['ORDINAL_COL_TRANSFORMER'][cfg['PATHS']['ORDINAL_COL_TRANSFORMER'].index('/')+1:]
cfg['PATHS']['OHE_COL_TRANSFORMER_MV'] = args.datadir + \
                                         cfg['PATHS']['OHE_COL_TRANSFORMER_MV'][cfg['PATHS']['OHE_COL_TRANSFORMER_MV'].index('/')+1:]
cfg['PATHS']['OHE_COL_TRANSFORMER_SV'] = args.datadir + \
                                         cfg['PATHS']['OHE_COL_TRANSFORMER_SV'][cfg['PATHS']['OHE_COL_TRANSFORMER_SV'].index('/')+1:]

# Create path for preprocessed data if it doesn't already exist on the blob
if not os.path.exists(args.preprocesseddir):
    os.makedirs(args.preprocesseddir)
if not os.path.exists(args.datadir + cfg['PATHS']['DATA_INFO'].split('/')[-2]):
    os.makedirs(args.datadir + cfg['PATHS']['DATA_INFO'].split('/')[-2])
if not os.path.exists(args.datadir + cfg['PATHS']['ORDINAL_COL_TRANSFORMER'].split('/')[-2]):
    os.makedirs(args.datadir + cfg['PATHS']['ORDINAL_COL_TRANSFORMER'].split('/')[-2])


# Run preprocessing.
preprocessed_data = preprocess(config=cfg, n_weeks=None, include_gt=True, calculate_gt=True,
                               classify_cat_feats=True, load_ct=False)