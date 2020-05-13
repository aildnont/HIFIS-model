import pandas as pd
import yaml
import os
import dill
import numpy as np
import scipy as sp
from tqdm import tqdm
from datetime import datetime
from sklearn.externals.joblib import load
from tensorflow.keras.models import load_model
from src.data.preprocess import preprocess
from src.interpretability.lime_explain import predict_and_explain, predict_instance

def predict_and_explain_set(cfg=None, data_path=None, save_results=True, give_explanations=True, include_feat_values=True,
                            processed_df=None):
    '''
    Preprocess a raw dataset. Then get model predictions and corresponding LIME explanations.
    :param cfg: Custom config object
    :param data_path: Path to look for raw data
    :param save_results: Flag specifying whether to save the prediction results to disk
    :param give_explanations: Flag specifying whether to provide LIME explanations with predictions spreadsheet
    :param include_feat_values: Flag specifying whether to include client feature values with predictions spreadsheet
    :param processed_df: Dataframe of preprocessed data. data_path will be ignored if passed.
    :return: Dataframe of prediction results, including explanations.
    '''

    # Load project config data
    if cfg is None:
        cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

    # Load data transformers
    scaler_ct = load(cfg['PATHS']['SCALER_COL_TRANSFORMER'])
    ohe_ct_sv = load(cfg['PATHS']['OHE_COL_TRANSFORMER_SV'])

    # Get preprocessed data
    if processed_df is not None:
        df = processed_df
        client_ids = np.array(df.pop('ClientID'))
    else:
        if data_path is None:
            data_path = cfg['PATHS']['RAW_DATA']

        # Preprocess the data, using pre-existing sklearn transformers and classification of categorical features.
        df = preprocess(n_weeks=0, include_gt=False, calculate_gt=False, classify_cat_feats=False, load_ct=True,
                          data_path=data_path)
        client_ids = np.array(df.index)

    # Ensure DataFrame does not contain ground truth (could happen if custom preprocessed data is passed)
    if 'GroundTruth' in df.columns:
        df.drop('GroundTruth', axis=1, inplace=True)

    # Load feature mapping information (from preprocessing)
    data_info = yaml.full_load(open(cfg['PATHS']['DATA_INFO'], 'r'))

    # Convert dataset to numpy array
    X = np.array(df)

    # Restore the model and LIME explainer from their respective serializations
    explainer = dill.load(open(cfg['PATHS']['LIME_EXPLAINER'], 'rb'))
    model = load_model(cfg['PATHS']['MODEL_TO_LOAD'], compile=False)

    # Get the predictive horizon that this model was trained on. It's embedded within the model name.
    n_weeks = int(model._name.split('_')[1].split('-')[0])

    # Load LIME and prediction constants from config
    NUM_SAMPLES = cfg['LIME']['NUM_SAMPLES']
    NUM_FEATURES = cfg['LIME']['NUM_FEATURES']
    THRESHOLD = cfg['PREDICTION']['THRESHOLD']
    CLASS_NAMES = cfg['PREDICTION']['CLASS_NAMES']

    # Define column names of the DataFrame representing the prediction results
    col_names = ['ClientID', 'Predictive Horizon [weeks]', 'At risk of chronic homelessness',
                 'Probability of chronic homelessness [%]']

    # Add columns for client explanation
    if give_explanations:
        for i in range(NUM_FEATURES):
            col_names.extend(['Explanation ' + str(i+1), 'Weight ' + str(i+1)])

    # Add columns for client feature values
    if include_feat_values:
        col_names.extend(list(df.columns))
    rows = []

    # Predict and explain all items in dataset
    print('Predicting and explaining examples.')
    for i in tqdm(X.shape[0]):

        # Predict this example
        x = np.expand_dims(X[i], axis=0)
        y = np.squeeze(predict_instance(x, model, ohe_ct_sv, scaler_ct).T, axis=1)  # Predict example
        prediction = 1 if y[1] >= THRESHOLD else 0  # Model's classification
        predicted_class = CLASS_NAMES[prediction]
        row = [client_ids[i], n_weeks, predicted_class, y[1] * 100]

        # Explain this prediction
        if give_explanations:
            x = sp.sparse.csr_matrix(X[i])
            explanation = predict_and_explain(x, model, explainer, ohe_ct_sv, scaler_ct, NUM_FEATURES, NUM_SAMPLES)
            exp_tuples = explanation.as_list()
            for exp_tuple in exp_tuples:
                row.extend(list(exp_tuple))
            if len(exp_tuples) < NUM_FEATURES:
                row.extend([''] * (2 * (NUM_FEATURES - len(exp_tuples))))   # Fill with empty space if explanation too small

        # Add client's feature values
        if include_feat_values:
            client_vals = list(df.loc[client_ids[i], :])
            for idx in data_info['SV_CAT_FEATURE_IDXS']:
                ordinal_encoded_val = int(client_vals[idx])
                client_vals[idx] = data_info['SV_CAT_VALUES'][idx][ordinal_encoded_val]
            row.extend(client_vals)

        rows.append(row)

    # Convert results to a Pandas dataframe and save
    results_df = pd.DataFrame(rows, columns=col_names)
    if save_results:
        results_path = cfg['PATHS']['BATCH_PREDICTIONS'] + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
        results_df.to_csv(results_path, columns=col_names, index_label=False, index=False)
    return results_df


def trending_prediction(data_path=None):
    '''
    Gets predictions and explanations from the provided raw data and adds a timestamp. Concatenates results with those
    of previous predictions on same client. Running this function periodically enables trend analysis for clients as the
    data changes over time.
    :param data_path: Path of raw data
    '''
    cfg = yaml.full_load(open("./config.yml", 'r'))
    trending_pred_path = cfg['PATHS']['TRENDING_PREDICTIONS']

    # Get model predictions and explanations.
    results_df = predict_and_explain_set(data_path=data_path, save_results=False)
    results_df.insert(0, 'Timestamp', pd.to_datetime(datetime.today()))     # Add a timestamp to these results
    col_names = list(results_df.columns)

    # If previous prediction file exists, load it and append the predictions we just made.
    if os.path.exists(trending_pred_path):
        prev_results_df = pd.read_csv(trending_pred_path)
        results_df = pd.concat([prev_results_df, results_df], axis=0, sort=False)

    # Save the updated trend analysis prediction spreadsheet
    results_df.to_csv(trending_pred_path, columns=col_names, index_label=False, index=False)
    return


if __name__ == '__main__':
    #results = predict_and_explain_set(data_path=None, save_results=True, give_explanations=True, include_feat_values=True)
    #trending_prediction(data_path=None)
    cfg = yaml.full_load(open("./config.yml", 'r'))
    df = pd.read_csv(cfg['PATHS']['PROCESSED_DATA'])
    results_df = predict_and_explain_set(cfg=None, data_path=None, save_results=True, give_explanations=True,
                            include_feat_values=True,
                            processed_df=df)