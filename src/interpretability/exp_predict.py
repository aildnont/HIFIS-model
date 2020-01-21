import pandas as pd
import yaml
import os
import numpy as np
import lime
import lime.lime_tabular
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import load
from tensorflow.keras.models import load_model

def predict(client_data):
    return

def predict_and_explain(x, exp, sv_cat_features, idx):
    for feature in sv_cat_features:
        r = 0
    return

# Load project config data
input_stream = open(os.getcwd() + "/config.yml", 'r')
cfg = yaml.full_load(input_stream)

# Load feature information
input_stream = open(os.getcwd() + cfg['PATHS']['INTERPRETABILITY'], 'r')
cfg_feats = yaml.full_load(input_stream)
mv_cat_features = cfg_feats['MV_CAT_FEATURES']
sv_cat_features = cfg_feats['SV_CAT_FEATURES']
noncat_features = cfg_feats['NON_CAT_FEATURES']
sv_cat_feature_idxs = cfg_feats['SV_CAT_FEATURE_IDXS']
sv_cat_values = cfg_feats['SV_CAT_VALUES']
vec_sv_cat_features = cfg_feats['VEC_SV_CAT_FEATURES']

# Load train and test sets
train_df = pd.read_csv(cfg['PATHS']['TRAIN_SET'])
test_df = pd.read_csv(cfg['PATHS']['TEST_SET'])

# Get ground truth values
Y_train = np.array(train_df.pop('GroundTruth'))
Y_test = np.array(test_df.pop('GroundTruth'))

# Scale train and test set values
scaler = load(cfg['PATHS']['STD_SCALER'])
train_df[noncat_features] = scaler.transform(train_df[noncat_features])
test_df[noncat_features] = scaler.transform(test_df[noncat_features])

# Get list of client IDs
train_client_ids = train_df.pop('ClientID')
test_client_ids = test_df.pop('ClientID')

# Convert datasets to numpy arrays
X_train = np.array(train_df)
X_test = np.array(test_df)

# Define the LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=train_df.columns,
                                                   class_names=['0', '1'], categorical_features=sv_cat_feature_idxs,
                                                   categorical_names=sv_cat_values, kernel_width=3)

# Load trained model's weights
model = load_model(cfg['PATHS']['MODEL_WEIGHTS'])

i = 0
predict_and_explain(X_test[i], explainer, sv_cat_features, i)

