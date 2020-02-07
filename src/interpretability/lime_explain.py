import pandas as pd
import yaml
import os
import datetime
import dill
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.externals.joblib import load
from tensorflow.keras.models import load_model
from src.visualization.visualize import visualize_explanation, visualize_avg_explanations
from src.custom.metrics import F1Score

def predict_instance(x, model, ohe_ct_sv, scaler_ct):
    '''
    Helper function for LIME explainer. Runs model prediction on perturbations of the example.
    :param x: List of perturbed examples from an example
    :param model: Keras model
    :param ohe_ct_sv: A column transformer for one hot encoding single-valued categorical features
    :param: scaler_ct: A column transformer for scaling numerical features
    :return: A numpy array constituting a list of class probabilities for each predicted perturbation
    '''
    x_ohe = ohe_ct_sv.transform(x)      # One hot encode the single-valued categorical features
    x_ohe = scaler_ct.transform(x_ohe)
    y = model.predict(x_ohe)  # Run prediction on the perturbations
    probs = np.concatenate([1.0 - y, y], axis=1)  # Compute class probabilities from the output of the model
    return probs

def predict_and_explain(x, model, exp, ohe_ct_sv, scaler_ct, num_features, num_samples):
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
        probs = predict_instance(x, model, ohe_ct_sv, scaler_ct)
        return probs

    # Generate explanation for the example
    explanation = exp.explain_instance(x, predict, num_features=num_features, num_samples=num_samples)
    return explanation

def lime_experiment(X_test, Y_test, model, exp, threshold, ohe_ct, scaler_ct, num_features, num_samples, file_path, all=False):
    '''
    Conduct an experiment to assess the predictions of all predicted positive and some negative cases in the test set.
    :param X_test: Numpy array of the test set
    :param Y_test: Pandas dataframe consisting of ClientIDs and corresponding ground truths
    :param model: Keras model
    :param exp: LimeTabularExplainer object
    :param threshold: Prediction threshold
    :param ohe_ct: A column transformer for one hot encoding single-valued categorical features
    :param: scaler_ct: A column transformer for scaling numerical features
    :param num_features: # of features to use in explanation
    :param num_samples: # of times to perturb the example to be explained
    :param file_path: Path at which to save experiment results
    '''
    run_start = datetime.datetime.today()    # Record start time of experiment
    NEG_EXP_PERIOD = 1      # We will explain positive to negative predicted examples at this ratio
    pos_exp_counter = 0     # Keeps track of how many positive examples are explained in a row

    # Define column names of the DataFrame representing the results
    col_names = ['ClientID', 'GroundTruth', 'Prediction', 'Classification', 'p(neg)', 'p(pos)']
    for i in range(num_features):
        col_names.extend(['Exp_' + str(i), 'Weight_' + str(i)])

    # Make predictions on the test set. Explain every positive prediction and some negative predictions
    rows = []
    for i in range(X_test.shape[0]):
        x = np.expand_dims(X_test[i], axis=0)
        y = np.squeeze(predict_instance(x, model, ohe_ct, scaler_ct).T, axis=1)     # Predict example
        pred = 1 if y[1] >= threshold else 0        # Model's classification
        client_id = Y_test.index[i]
        gt = Y_test.loc[client_id, 'GroundTruth']   # Ground truth

        # Determine classification of this example compared to ground truth
        if pred == 1 and gt == 1:
            classification = 'TP'
        elif pred == 1 and gt == 0:
            classification = 'FP'
        elif pred == 0 and gt == 1:
            classification = 'FN'
        else:
            classification = 'TN'

        # Explain this example.
        if (pred == 1) or (gt == 1) or (pos_exp_counter >= NEG_EXP_PERIOD) or all:
            print('Explaining test example ', i, '/', X_test.shape[0])
            row = [client_id, gt, pred, classification, y[0], y[1]]

            # Explain this prediction
            explanation = predict_and_explain(X_test[i], model, exp, ohe_ct, scaler_ct, num_features, num_samples)
            exp_tuples = explanation.as_list()
            for exp_tuple in exp_tuples:
                row.extend(list(exp_tuple))
            rows.append(row)

            # Ensure a negative prediction is explained for every NEG_EXP_PERIOD positive predictions
            if pred == 1:
                pos_exp_counter += 1
            else:
                pos_exp_counter = 0

    # Convert results to a Pandas dataframe and save
    results_df = pd.DataFrame(rows, columns=col_names).set_index('ClientID')
    results_df.to_csv(file_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv')
    print("Runtime = ", ((datetime.datetime.today() - run_start).seconds / 60), " min")  # Print runtime of experiment
    return results_df

def run_lime():
    # Load relevant constants from project config file
    input_stream = open(os.getcwd() + "/config.yml", 'r')
    cfg = yaml.full_load(input_stream)
    NUM_SAMPLES = cfg['LIME']['NUM_SAMPLES']
    NUM_FEATURES = cfg['LIME']['NUM_FEATURES']
    KERNEL_WIDTH = cfg['LIME']['KERNEL_WIDTH']
    FEATURE_SELECTION = cfg['LIME']['FEATURE_SELECTION']
    FILE_PATH = cfg['PATHS']['LIME_EXPERIMENT']

    # Load feature information
    input_stream = open(os.getcwd() + cfg['PATHS']['DATA_INFO'], 'r')
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
    ohe_ct_sv = load(cfg['PATHS']['OHE_COL_TRANSFORMER_SV'])

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
    dill.dump(explainer, open(cfg['PATHS']['LIME_EXPLAINER'], 'wb'))    # Save

    # Load trained model's weights
    model = load_model(cfg['PATHS']['MODEL_WEIGHTS'])

    # Make a prediction and explain the rationale
    '''
    client_id = 76037
    i = Y_test.index.get_loc(client_id)
    explanation = predict_and_explain(X_test[i], model, explainer, ohe_ct_sv, scaler_ct, NUM_FEATURES, NUM_SAMPLES)
    visualize_explanation(explanation, client_id, Y_test.loc[client_id, 'GroundTruth'])
    '''

    '''
    results_df = lime_experiment(X_test, Y_test, model, explainer, ohe_ct_sv, scaler_ct, NUM_FEATURES, NUM_SAMPLES, FILE_PATH, all=False)
    visualize_avg_explanations(results_df, file_path=cfg['PATHS']['IMAGES'])
    '''

if __name__ == '__main__':
    results = run_lime()