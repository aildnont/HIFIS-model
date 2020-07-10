import os
import yaml
import pandas as pd
import argparse
import datetime
from azureml.core import Run
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from src.data.preprocess import preprocess

parser = argparse.ArgumentParser()
parser.add_argument('--rawdatadir', type=str, help="Raw HIFIS client data directory")
parser.add_argument('--inferencedir', type=str, help="directory containing all files necessary for inference")
parser.add_argument('--preprocessedoutputdir', type=str, help="intermediate serialized pipeline data")
args = parser.parse_args()
run = Run.get_context()
cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

# Modify paths in config file based the Azure datastore paths passed as arguments.
cfg = yaml.full_load(open("./config.yml", 'r'))  # Load config data
cfg['PATHS']['RAW_DATA'] = args.rawdatadir + '/' + cfg['PATHS']['RAW_DATA'].split('/')[-1]
cfg['PATHS']['RAW_SPDAT_DATA'] = args.rawdatadir + '/' + cfg['PATHS']['RAW_SPDAT_DATA'].split('/')[-1]
cfg['PATHS']['PROCESSED_DATA'] = args.preprocessedoutputdir + '/' + cfg['PATHS']['PROCESSED_DATA'].split('/')[-1]
cfg['PATHS']['PROCESSED_OHE_DATA'] = args.preprocessedoutputdir + '/' + cfg['PATHS']['PROCESSED_OHE_DATA'].split('/')[-1]
cfg['PATHS']['TRAIN_SET'] = args.preprocessedoutputdir + '/' + cfg['PATHS']['TRAIN_SET'].split('/')[-1]
cfg['PATHS']['TEST_SET'] = args.preprocessedoutputdir + '/' + cfg['PATHS']['TEST_SET'].split('/')[-1]
cfg['PATHS']['GROUND_TRUTH'] = args.preprocessedoutputdir + '/' + cfg['PATHS']['GROUND_TRUTH'].split('/')[-1]

# A few checks to screen for problems with SQL query that retrieves HIFIS data. In these cases, send alert email.
raw_df = pd.read_csv(cfg['PATHS']['RAW_DATA'], encoding="ISO-8859-1", low_memory=False)

# Load meta-info from the last retrieved snapshot of raw HIFIS data
raw_data_info_path = args.inferencedir + '/raw_data_info.yml'
if os.path.exists(raw_data_info_path):
    raw_data_info = yaml.full_load(open(raw_data_info_path, 'r'))  # Load config data
else:
    raw_data_info = {'N_ROWS': raw_df.shape[0], 'N_COLS': raw_df.shape[1]}

check_date = datetime.datetime.today() - datetime.timedelta(days=7)    # 1 week ago from today
raw_df['DateStart'] = pd.to_datetime(raw_df['DateStart'], errors='coerce')
recent_df = raw_df[raw_df['DateStart'] > check_date]               # Get rows with service occurring in last week
failure_msg = ''
if recent_df.shape[0] == 0:
    failure_msg += 'HIFIS_Clients.csv did not contain any service entries with start dates within the last week.\n'
if raw_df.shape[0] < raw_data_info['N_ROWS']:
    failure_msg += 'HIFIS_Clients.csv contains less rows than it did last week. Last week it contained ' + str(raw_data_info['N_ROWS']) + ' rows and today it contains ' + str(raw_df.shape[0]) + ' rows.\n'
if raw_df.shape[1] < raw_data_info['N_COLS']:
    failure_msg += 'HIFIS_Clients.csv contains less columns than it did last week. Last week it contained ' + str(raw_data_info['N_COLS']) + ' columns and today it contains ' + str(raw_df.shape[1]) + ' columns.\n'
print("Current shape: ", raw_df.shape)

# Update raw data meta-info file
raw_data_info['N_ROWS'] = raw_df.shape[0]
raw_data_info['N_COLS'] = raw_df.shape[1]
with open(raw_data_info_path, 'w') as file:
    raw_data_info_doc = yaml.dump(raw_data_info, file)

# If necessary, send alert email and trigger pipeline run failure.
if len(failure_msg) > 0:
    failure_msg += 'Azure machine learning pipeline run was cancelled. Please check the SQL query.'
    cfg_private = yaml.full_load(open("./config-private.yml", 'r'))  # Load private config data
    message = Mail(from_email='HIFISModelAlerts@no-reply.ca', to_emails=cfg_private['EMAIL']['TO_EMAILS'],
                   subject='HIFIS Raw Client Data', html_content=failure_msg)
    for email_address in cfg_private['EMAIL']['CC_EMAILS']:
        message.add_cc(email_address)
    try:
        sg = SendGridAPIClient('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')
        response = sg.send(message)
    except Exception as e:
        print(str(e.body))
    raise Exception(failure_msg)


# Create path for preproces step output data if it doesn't already exist on the blob.
if not os.path.exists(args.preprocessedoutputdir):
    os.makedirs(args.preprocessedoutputdir)

# Run preprocessing.
if os.getenv("AML_PARAMETER_PIPELINE") == 'train':
    cfg['PATHS']['DATA_INFO'] = args.preprocessedoutputdir + '/' + cfg['PATHS']['DATA_INFO'].split('/')[-1]
    cfg['PATHS']['ORDINAL_COL_TRANSFORMER'] = args.preprocessedoutputdir + '/' + cfg['PATHS']['ORDINAL_COL_TRANSFORMER'].split('/')[-1]
    cfg['PATHS']['OHE_COL_TRANSFORMER_MV'] = args.preprocessedoutputdir + '/' + cfg['PATHS']['OHE_COL_TRANSFORMER_MV'].split('/')[-1]
    cfg['PATHS']['OHE_COL_TRANSFORMER_SV'] = args.preprocessedoutputdir + '/' + cfg['PATHS']['OHE_COL_TRANSFORMER_SV'].split('/')[-1]
    preprocessed_data = preprocess(cfg=cfg, n_weeks=None, include_gt=True, calculate_gt=True, classify_cat_feats=True,
                                   load_ct=False)   # Preprocessing for training
else:
    cfg['PATHS']['DATA_INFO'] = args.inferencedir + cfg['PATHS']['DATA_INFO'].split('/')[-1]
    cfg['PATHS']['OHE_COL_TRANSFORMER_SV'] = args.inferencedir + cfg['PATHS']['OHE_COL_TRANSFORMER_SV'].split('/')[-1]
    cfg['PATHS']['ORDINAL_COL_TRANSFORMER'] = args.inferencedir + cfg['PATHS']['ORDINAL_COL_TRANSFORMER'].split('/')[-1]
    cfg['PATHS']['OHE_COL_TRANSFORMER_MV'] = args.inferencedir + cfg['PATHS']['OHE_COL_TRANSFORMER_MV'].split('/')[-1]
    preprocessed_data = preprocess(cfg=cfg, n_weeks=0, include_gt=False, calculate_gt=False, classify_cat_feats=False,
                                   load_ct=True)    # Preprocessing for inference
    print("SHAPE", preprocessed_data.shape)
    preprocessed_data.to_csv(cfg['PATHS']['PROCESSED_DATA'], sep=',', header=True)

