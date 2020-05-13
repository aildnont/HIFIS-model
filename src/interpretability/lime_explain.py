import pandas as pd
import yaml
import os
import datetime
import dill
import collections
import numpy as np
import scipy as sp
from src.interpretability.lime_tabular import LimeTabularExplainer
from src.interpretability.submodular_pick import SubmodularPick
#from lime.lime_tabular import LimeTabularExplainer
#from lime.submodular_pick import SubmodularPick
from joblib import load
from tensorflow.keras.models import load_model
from src.visualization.visualize import visualize_explanation, visualize_avg_explanations, visualize_submodular_pick

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
        if sp.sparse.issparse(x):
            x = x.toarray()
        probs = predict_instance(x, model, ohe_ct_sv, scaler_ct)
        return probs

    # Generate explanation for the example
    explanation = exp.explain_instance(x, predict, num_features=num_features, num_samples=num_samples)
    return explanation


def setup_lime(cfg=None):
    '''
    Load relevant information and create a LIME Explainer
    :param: cfg: custom config object
    :return: dict containing important information and objects for explanation experiments
    '''

    # Load relevant constants from project config file if custsom config object not supplied
    if cfg is None:
        cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
    lime_dict = {}
    lime_dict['NUM_SAMPLES'] = cfg['LIME']['NUM_SAMPLES']
    lime_dict['NUM_FEATURES'] = cfg['LIME']['NUM_FEATURES']
    lime_dict['SAMPLE_FRACTION'] = cfg['LIME']['SP']['SAMPLE_FRACTION']
    lime_dict['FILE_PATH'] = cfg['PATHS']['LIME_EXPERIMENT']
    lime_dict['IMG_PATH'] = cfg['PATHS']['IMAGES']
    lime_dict['SP_FILE_PATH'] = cfg['PATHS']['LIME_SUBMODULAR_PICK']
    lime_dict['NUM_EXPLANATIONS'] = cfg['LIME']['SP']['NUM_EXPLANATIONS']
    lime_dict['PRED_THRESHOLD'] = cfg['PREDICTION']['THRESHOLD']
    KERNEL_WIDTH = cfg['LIME']['KERNEL_WIDTH']
    FEATURE_SELECTION = cfg['LIME']['FEATURE_SELECTION']

    # Load feature information
    input_stream = open(cfg['PATHS']['DATA_INFO'], 'r')
    cfg_feats = yaml.full_load(input_stream)
    noncat_features = cfg_feats['NON_CAT_FEATURES']
    sv_cat_values = cfg_feats['SV_CAT_VALUES']

    # Load train and test sets
    train_df = pd.read_csv(cfg['PATHS']['TRAIN_SET'])
    test_df = pd.read_csv(cfg['PATHS']['TEST_SET'])

    # Get client IDs and corresponding ground truths
    Y_train = pd.concat([train_df.pop(y) for y in ['ClientID', 'GroundTruth']], axis=1).set_index('ClientID')
    lime_dict['Y_TEST'] = pd.concat([test_df.pop(y) for y in ['ClientID', 'GroundTruth']], axis=1).set_index('ClientID')

    # Load data transformers
    lime_dict['SCALER_CT'] = load(cfg['PATHS']['SCALER_COL_TRANSFORMER'])
    lime_dict['OHE_CT_SV'] = load(cfg['PATHS']['OHE_COL_TRANSFORMER_SV'])

    # Get indices of categorical and noncategorical featuress
    noncat_feat_idxs = [test_df.columns.get_loc(c) for c in noncat_features if c in test_df]
    cat_feat_idxs = [i for i in range(len(test_df.columns)) if i not in noncat_feat_idxs]

    # Convert datasets to numpy arrays
    lime_dict['X_TRAIN'] = np.array(train_df)
    lime_dict['X_TEST'] = np.array(test_df)

    # Define the LIME explainer
    train_labels = Y_train['GroundTruth'].to_numpy()
    feature_names = train_df.columns.tolist()

    # Get training data stats
    training_data_stats = {'means': {}, 'mins': {}, 'maxs': {}, 'stds': {}, 'feature_values': {}, 'feature_frequencies': {}}
    for i in range(len(feature_names)):
        training_data_stats['means'][i] = np.mean(lime_dict['X_TRAIN'][:,i])
        training_data_stats['mins'][i] = np.min(lime_dict['X_TRAIN'][:, i])
        training_data_stats['maxs'][i] = np.max(lime_dict['X_TRAIN'][:, i])
        training_data_stats['stds'][i] = np.std(lime_dict['X_TRAIN'][:, i])
        values, frequencies = map(list, zip(*(sorted(collections.Counter(lime_dict['X_TRAIN'][:, i]).items()))))
        training_data_stats['feature_values'][i] = values
        training_data_stats['feature_frequencies'][i] = frequencies

    # Convert to sparse matrices
    lime_dict['X_TRAIN'] = sp.sparse.csr_matrix(lime_dict['X_TRAIN'])
    lime_dict['X_TEST'] = sp.sparse.csr_matrix(lime_dict['X_TEST'])

    lime_dict['EXPLAINER'] = LimeTabularExplainer(lime_dict['X_TRAIN'], feature_names=feature_names, class_names=['0', '1'],
                                    categorical_features=cat_feat_idxs, categorical_names=sv_cat_values, training_labels=train_labels,
                                    kernel_width=KERNEL_WIDTH, feature_selection=FEATURE_SELECTION, discretizer='decile',
                                    discretize_continuous=True)
    dill.dump(lime_dict['EXPLAINER'], open(cfg['PATHS']['LIME_EXPLAINER'], 'wb'))    # Serialize the explainer

    # Load trained model's weights
    lime_dict['MODEL'] = load_model(cfg['PATHS']['MODEL_TO_LOAD'], compile=False)
    return lime_dict


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
    row_len = len(col_names) + 2 * num_features
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
            if len(row) == row_len:
                rows.append(row)
            else:
                print("Unusual amount of explanations for test example ", i, ". Length of row is ", len(row))

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


def submodular_pick(lime_dict, file_path=None):
    '''
    Perform submodular pick on the training set to approximate global explanations for the model.
    :param lime_dict: dict containing important information and objects for explanation experiments
    :param file_path: Path at which to save the image summarizing the submodular pick
    '''

    def predict_example(x):
        '''
        Helper function for LIME explainer. Runs model prediction on perturbations of the example.
        :param x: List of perturbed examples from an example
        :return: A numpy array constituting a list of class probabilities for each predicted perturbation
        '''
        if sp.sparse.issparse(x):
            x = x.toarray()
        probs = predict_instance(x, lime_dict['MODEL'], lime_dict['OHE_CT_SV'], lime_dict['SCALER_CT'])
        return probs

    start_time = datetime.datetime.now()

    # Set sample size, ensuring that it isn't larger than size of training set.
    sample_size = int(lime_dict['SAMPLE_FRACTION'] * lime_dict['X_TRAIN'].shape[0])
    if (sample_size == 'all') or (sample_size > lime_dict['X_TRAIN'].shape[0]):
        sample_size = lime_dict['X_TRAIN'].shape[0]

    # Perform a submodular pick of explanations of uniformly sampled examples from the training set
    submod_picker = SubmodularPick(lime_dict['EXPLAINER'], lime_dict['X_TRAIN'], predict_example,
                                   sample_size=sample_size, num_features=lime_dict['NUM_FEATURES'],
                                   num_exps_desired=lime_dict['NUM_EXPLANATIONS'], top_labels=None,
                                   num_samples=lime_dict['NUM_SAMPLES'])
    print("Submodular pick time = " + str((datetime.datetime.now() - start_time).total_seconds() / 60) + " minutes")

    # Assemble all explanations in a DataFrame
    W = pd.DataFrame([dict(exp.as_list()) for exp in submod_picker.sp_explanations])

    # Calculate mean of explanations encountered across the picked examples. Ignore nan values.
    W_avg = W.mean(skipna=True).T

    # Save average explanations from submodular pick to .csv file
    W_avg_df = W_avg.to_frame()
    W_avg_df.reset_index(level=0, inplace=True)
    W_avg_df.columns = ['Explanation', 'Avg Weight']
    W_avg_df["Abs Weight"] = np.abs(W_avg_df['Avg Weight'])
    W_avg_df.sort_values('Abs Weight', inplace=True, ascending=False)
    W_avg_df.drop('Abs Weight', inplace=True, axis=1)
    sp_file_path = lime_dict['SP_FILE_PATH']
    W_avg_df.insert(0, 'Timestamp', pd.to_datetime(datetime.datetime.now()))  # Add a timestamp to these results
    if os.path.exists(sp_file_path):
        prev_W_avg_df = pd.read_csv(sp_file_path)
        W_avg_df = pd.concat([prev_W_avg_df, W_avg_df], axis=0)    # Concatenate these results with previous results
    W_avg_df.to_csv(sp_file_path, index_label=False, index=False)

    # Visualize the the average explanations
    sample_fraction = sample_size / lime_dict['X_TRAIN'].shape[0]
    if file_path is None:
        file_path = lime_dict['IMG_PATH'] + 'LIME_Submodular_Pick_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png'
    visualize_submodular_pick(W_avg, sample_fraction, file_path=file_path)
    return


def explain_single_client(lime_dict, client_id):
    '''
    # Make a prediction and explain the rationale
    :param lime_dict: dict containing important information and objects for explanation experiments
    :param client_id: Client to predict and explain
    '''
    i = lime_dict['Y_TEST'].index.get_loc(client_id)
    start_time = datetime.datetime.now()
    explanation = predict_and_explain(lime_dict['X_TEST'][i], lime_dict['MODEL'], lime_dict['EXPLAINER'],
                                      lime_dict['OHE_CT_SV'], lime_dict['SCALER_CT'], lime_dict['NUM_FEATURES'],
                                      lime_dict['NUM_SAMPLES'])
    print("Explanation time = " + str((datetime.datetime.now() - start_time).total_seconds()) + " seconds")
    fig = visualize_explanation(explanation, client_id, lime_dict['Y_TEST'].loc[client_id, 'GroundTruth'],
                          file_path=lime_dict['IMG_PATH'])
    return

def run_lime_experiment_and_visualize(lime_dict):
    '''
    Run LIME experiment on test set and visualize average explanations
    :param lime_dict: dict containing important information and objects for explanation experiments
    '''
    results_df = lime_experiment(lime_dict['X_TEST'], lime_dict['Y_TEST'], lime_dict['MODEL'], lime_dict['EXPLAINER'],
                              lime_dict['PRED_THRESHOLD'], lime_dict['OHE_CT_SV'], lime_dict['SCALER_CT'],
                              lime_dict['NUM_FEATURES'], lime_dict['NUM_SAMPLES'], lime_dict['FILE_PATH'], all=True)
    sample_fraction = results_df.shape[0] / lime_dict['X_TEST'].shape[0]
    visualize_avg_explanations(results_df, sample_fraction, file_path=lime_dict['IMG_PATH'])
    return

if __name__ == '__main__':
    lime_dict = setup_lime()
    explain_single_client(lime_dict, 86182)    # Replace with Client ID from a client in test set
    #run_lime_experiment_and_visualize(lime_dict)
    #submodular_pick(lime_dict)
