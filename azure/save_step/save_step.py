import os
import yaml
import argparse
import datetime
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--preprocessedoutputdir', type=str, help="intermediate preprocessed pipeline data directory")
parser.add_argument('--trainoutputdir', type=str, help="intermediate training pipeline data directory")
parser.add_argument('--interpretabilityoutputdir', type=str, help="intermediate interpretability pipeline data directory")
parser.add_argument('--outputsdir', type=str, help="persistent outputs directory on blob")
args = parser.parse_args()
cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

# Get paths of data on intermediate pipeline storage
cfg = yaml.full_load(open("./config.yml", 'r'))  # Load config data
train_set_path = args.preprocessedoutputdir + '/' + cfg['PATHS']['TRAIN_SET'].split('/')[-1]
test_set_path = args.preprocessedoutputdir + '/' + cfg['PATHS']['TEST_SET'].split('/')[-1]
data_info_path = args.preprocessedoutputdir + '/' + cfg['PATHS']['DATA_INFO'].split('/')[-1]
ordinal_col_transformer_path = args.preprocessedoutputdir + '/' + cfg['PATHS']['ORDINAL_COL_TRANSFORMER'].split('/')[-1]
ohe_col_transformer_mv_path = args.preprocessedoutputdir + '/' + cfg['PATHS']['OHE_COL_TRANSFORMER_MV'].split('/')[-1]
ohe_col_transformer_sv_path = args.preprocessedoutputdir + '/' + cfg['PATHS']['OHE_COL_TRANSFORMER_SV'].split('/')[-1]
scaler_col_transformer_path = args.trainoutputdir + '/' + cfg['PATHS']['SCALER_COL_TRANSFORMER'].split('/')[-1]
model_to_load_path = args.trainoutputdir + '/' + cfg['PATHS']['MODEL_TO_LOAD'].split('/')[-1]
logs_path = args.trainoutputdir + '/logs'
multi_train_test_metrics_path = args.trainoutputdir + '/' + cfg['PATHS']['MULTI_TRAIN_TEST_METRICS'].split('/')[-1]
lime_explainer_path = args.interpretabilityoutputdir + '/' + cfg['PATHS']['LIME_EXPLAINER'].split('/')[-1]
lime_submodular_pick_path = args.interpretabilityoutputdir + '/' + cfg['PATHS']['LIME_SUBMODULAR_PICK'].split('/')[-1]
submod_pick_image_path = args.interpretabilityoutputdir + '/' + cfg['PATHS']['IMAGES'].split('/')[-1] + '/submodular_pick.png'

# Build destination paths in output folder on blob datastore
destination_dir = args.outputsdir + cur_date + '/'
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)
business_outputs_dir = destination_dir + 'business_outputs/'
if not os.path.exists(business_outputs_dir):
    os.makedirs(business_outputs_dir)

# Move all outputs from intermediate data to outputs folder on blob
shutil.copy(train_set_path, destination_dir)
shutil.copy(test_set_path, destination_dir)
shutil.copy(data_info_path, destination_dir)
shutil.copy(ordinal_col_transformer_path, destination_dir)
shutil.copy(ohe_col_transformer_mv_path, destination_dir)
shutil.copy(ohe_col_transformer_sv_path, destination_dir)
shutil.copy(scaler_col_transformer_path, destination_dir)
shutil.copy(model_to_load_path, destination_dir)
shutil.copytree(logs_path, destination_dir + 'logs')
shutil.copy(multi_train_test_metrics_path, business_outputs_dir)
shutil.copy(lime_explainer_path, destination_dir)
shutil.copy(lime_submodular_pick_path, destination_dir)
shutil.copy(submod_pick_image_path, business_outputs_dir)