import pandas as pd
import yaml
import os
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.externals.joblib import load
from tensorflow.keras.models import load_model
from src.visualization.visualize import visualize_explanation


def predict_instance(x, model, ohe_ct, scaler_ct):
    '''
    Helper function for LIME explainer. Runs model prediction on perturbations of the example.
    :param x: List of perturbed examples from an example
    :param model: Keras model
    :param ohe_ct: A column transformer for one hot encoding single-valued categorical features
    :param: scaler_ct: A column transformer for scaling numerical features
    :return: A numpy array constituting a list of class probabilities for each predicted perturbation
    '''
    x = np.insert(x, x.shape[1], [1], axis=1)  # Insert dummy column where GroundTruth would be
    x_ohe = ohe_ct.transform(x)  # One hot encode the single-valued categorical features
    x_ohe = x_ohe[:, :-1]  # Remove the dummy column
    x_ohe = scaler_ct.transform(x_ohe)
    y = model.predict(x_ohe)  # Run prediction on the perturbations
    probs = np.concatenate([1.0 - y, y], axis=1)  # Compute class probabilities from the output of the model
    return probs

def predict_and_explain(x, model, exp, ohe_ct, scaler_ct, num_features, num_samples):
    '''
    Use the model to predict a single example and apply LIME to generate an explanation.
    :param x: An example (i.e. 1 client row)
    :param model: The trained neural network model
    :param exp: A LimeTabularExplainer object
    :param ohe_ct: A column transformer for one hot encoding single-valued categorical features
    :param scaler_ct: A column transformer for scaling numerical features
    :param num_features: # of features to use in explanation
    :param num_samples: # of times to perturb the example to be explained
    :return: The LIME explainer for the instance
    '''

    def predict(x):
        '''
        Helper function for LIME explainer. Runs model prediction on perturbations of the example.
        :param x: List of perturbed examples from an example
        :return: A numpy array constituting a list of class probabilities for each predicted perturbation
        '''
        probs = predict_instance(x, model, ohe_ct, scaler_ct)
        return probs

    # Generate explanation for the example
    explanation = exp.explain_instance(x, predict, num_features=num_features, num_samples=num_samples)
    return explanation

def lime_experiment(X_test, Y_test, model, exp, ohe_ct, scaler_ct, num_features, num_samples, file_path):
    '''
    Conduct an experiment to assess the predictions of all predicted positive and some negative cases in the test set.
    :param X_test: Numpy array of the test set
    :param Y_test: Pandas dataframe consisting of ClientIDs and corresponding ground truths
    :param model: Keras model
    :param exp: LimeTabularExplainer object
    :param ohe_ct: A column transformer for one hot encoding single-valued categorical features
    :param: scaler_ct: A column transformer for scaling numerical features
    :param num_features: # of features to use in explanation
    :param num_samples: # of times to perturb the example to be explained
    :param file_path: Path at which to save experiment results
    '''
    NEG_EXP_PERIOD = 5      # We will explain positive to negative predicted examples at this ratio
    THRESHOLD = 0.5         # Classification threshold
    pos_exp_counter = 0     # Keeps track of how many positive examples are explained in a row

    # Define column names of the DataFrame representing the results
    col_names = ['ClientID', 'GroundTruth', 'p(neg)', 'p(pos)']
    col_names.extend(['Exp_' + str(i) for i in range(num_features)])

    # Make predictions on the test set. Explain every positive prediction and some negative predictions
    rows = []
    for i in range(X_test.shape[0]):
        x = np.expand_dims(X_test[i], axis=0)
        y = np.squeeze(predict_instance(x, model, ohe_ct, scaler_ct).T, axis=1)     # Predict example
        if y[1] >= THRESHOLD or pos_exp_counter >= NEG_EXP_PERIOD:
            print('Explaining test example ', i, '/', X_test.shape[0])
            client_id = Y_test.index[i]
            row = [client_id, Y_test.loc[client_id, 'GroundTruth'], y[0], y[1]]

            # Explain this prediction
            explanation = predict_and_explain(X_test[i], model, exp, ohe_ct, scaler_ct, num_features, num_samples)
            row.extend(explanation.as_list())
            rows.append(row)

            # Ensure a negative prediction is explained for every NEG_EXP_PERIOD positive predictions
            if y[1] >= THRESHOLD:
                pos_exp_counter += 1
            else:
                pos_exp_counter = 0

    # Convert results to a Pandas dataframe and save
    results_df = pd.DataFrame(rows, columns=col_names)
    results_df.to_csv(file_path)
    return

# Load relevant constants from project config file
input_stream = open(os.getcwd() + "/config.yml", 'r')
cfg = yaml.full_load(input_stream)
NUM_SAMPLES = cfg['LIME']['NUM_SAMPLES']
NUM_FEATURES = cfg['LIME']['NUM_FEATURES']
KERNEL_WIDTH = cfg['LIME']['KERNEL_WIDTH']
FEATURE_SELECTION = cfg['LIME']['FEATURE_SELECTION']
FILE_PATH = cfg['PATHS']['LIME_EXPERIMENT']

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
scaler_ct = load(cfg['PATHS']['SCALER_COL_TRANSFORMER'])
ohe_ct = load(cfg['PATHS']['OHE_COL_TRANSFORMER'])

# Get indices of categorical and noncategorical featuress
noncat_feat_idxs = [test_df.columns.get_loc(c) for c in noncat_features if c in test_df]
cat_feat_idxs = [i for i in range(len(test_df.columns)) if i not in noncat_feat_idxs]

# Convert datasets to numpy arrays
X_train = np.array(train_df)
X_test = np.array(test_df)

# Define the LIME explainer
train_labels = Y_train['GroundTruth'].to_numpy()
explainer = LimeTabularExplainer(X_train, feature_names=train_df.columns, class_names=['0', '1'],
                                categorical_features=cat_feat_idxs, categorical_names=sv_cat_values, training_labels=train_labels,
                                kernel_width=KERNEL_WIDTH, feature_selection=FEATURE_SELECTION)

# Load trained model's weights
model = load_model(cfg['PATHS']['MODEL_WEIGHTS'])

# Make a prediction and explain the rationale
'''
client_id = Y_test.index[0]
i = Y_test.index.get_loc(client_id)
explanation = predict_and_explain(X_test[i], model, explainer, ohe_ct, scaler_ct, NUM_FEATURES, NUM_SAMPLES)
visualize_explanation(explanation, client_id, Y_test.loc[client_id, 'GroundTruth'])
'''

lime_experiment(X_test, Y_test, model, explainer, ohe_ct, scaler_ct, NUM_FEATURES, NUM_SAMPLES, FILE_PATH)
