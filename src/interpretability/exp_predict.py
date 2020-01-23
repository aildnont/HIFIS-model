import pandas as pd
import yaml
import os
import numpy as np
import lime
import lime.lime_tabular
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.externals.joblib import load
from tensorflow.keras.models import load_model
from src.visualization.visualize import visualize_explanation


def predict_and_explain(x, model, exp, ohe_col_transformer, scaler_col_transformer, noncat_feat_idxs, num_features):
    '''
    Use the model to predict a single example and apply LIME to generate an explanation.
    :param x: An example (i.e. 1 client row)
    :param model: The trained neural network model
    :param exp: A LimeTabularExplainer object
    :param sv_cat_features: List of single-valued categorical features
    :param ohe_col_transformer: A column transformer for one hot encoding single-valued categorical features
    :return: The LIME explainer for the instance
    '''

    def predict_instance(x):
        '''
        Helper function for LIME explainer. Runs model prediction on perturbations of the example.
        :param x: List of perturbed examples from an example
        :param x: number of top explainable features to report
        :return: A numpy array constituting a list of class probabilities for each predicted perturbation
        '''
        x = np.insert(x, x.shape[1], [1], axis=1)  # Insert dummy column where GroundTruth would be
        x_ohe = ohe_col_transformer.transform(x)    # One hot encode the single-valued categorical features
        x_ohe = x_ohe[:,:-1]    # Remove the dummy column
        x_ohe = scaler_col_transformer.transform(x_ohe)
        y = model.predict(x_ohe)    # Run prediction on the perturbations
        probs = np.concatenate([1.0 - y, y], axis=1)    # Compute class probabilities from the output of the model
        return probs

    explanation = exp.explain_instance(x, predict_instance, num_features=num_features)     # Generate explanation for the example
    return explanation

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

# Get client IDs and corresponding ground truths
Y_train = pd.concat([train_df.pop(x) for x in ['ClientID', 'GroundTruth']], axis=1).set_index('ClientID')
Y_test = pd.concat([test_df.pop(x) for x in ['ClientID', 'GroundTruth']], axis=1).set_index('ClientID')

# Load data transformers
scaler_col_transformer = load(cfg['PATHS']['SCALER_COL_TRANSFORMER'])
ohe_col_transformer = load(cfg['PATHS']['OHE_COL_TRANSFORMER'])

# Get indices of categorical and noncategorical featuress
noncat_feat_idxs = [test_df.columns.get_loc(c) for c in noncat_features if c in test_df]
cat_feat_idxs = [i for i in range(len(test_df.columns)) if i not in noncat_feat_idxs]

# Convert datasets to numpy arrays
X_train = np.array(train_df)
X_test = np.array(test_df)

# Define the LIME explainer
train_labels = Y_train['GroundTruth'].to_numpy()
explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=train_df.columns, class_names=['0', '1'],
                                                   categorical_features=cat_feat_idxs, categorical_names=sv_cat_values,
                                                   training_labels=train_labels)

# Load trained model's weights
model = load_model(cfg['PATHS']['MODEL_WEIGHTS'])

# Make a prediction and explain the rationale
client_id = 82644
i = Y_test.index.get_loc(client_id)
explanation = predict_and_explain(X_test[i], model, explainer, ohe_col_transformer, scaler_col_transformer, noncat_feat_idxs, 15)
visualize_explanation(explanation, client_id, Y_test.loc[client_id, 'GroundTruth'])
