import pandas as pd
import yaml
import os
import dill
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sklearn.externals.joblib import load
from tensorflow.keras.models import load_model
from src.data.preprocess import preprocess
from src.interpretability.lime_explain import predict_and_explain, predict_instance

def predict_and_explain_set(data_path=None, save_results=True, give_explanations=True):
    '''
    Preprocess a raw dataset. Then get model predictions and corresponding LIME explanations.
    :param data_path: Path at which to save results of this prediction
    :param save_results: Flag specifying whether to save the prediction results to disk
    :param give_explanations: Flag specifying whether to provide LIME explanations with predictions
    :return: Dataframe of prediction results, including explanations.
    '''

    # Load project config data
    input_stream = open(os.getcwd() + "/config.yml", 'r')
    cfg = yaml.full_load(input_stream)

    # Load data transformers
    scaler_ct = load(cfg['PATHS']['SCALER_COL_TRANSFORMER'])
    ohe_ct_sv = load(cfg['PATHS']['OHE_COL_TRANSFORMER_SV'])

    # Preprocess the data, using pre-existing sklearn transformers and classification of categorical features.
    if data_path is None:
        data_path = cfg['PATHS']['RAW_DATA']
    df = preprocess(n_weeks=0, include_gt=False, calculate_gt=False, classify_cat_feats=False, load_ct=True,
                      data_path=data_path)
    client_ids = np.array(df.index)     # index is ClientID

    # Convert dataset to numpy array
    X = np.array(df)

    # Restore the model and LIME explainer from their respective serializations
    explainer = dill.load(open(cfg['PATHS']['LIME_EXPLAINER'], 'rb'))
    model = load_model(cfg['PATHS']['MODEL_TO_LOAD'])

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
    if give_explanations:
        for i in range(NUM_FEATURES):
            col_names.extend(['Explanation ' + str(i+1), 'Weight ' + str(i+1)])
    rows = []

    # Predict and explain all items in dataset
    print('Predicting and explaining examples.')
    for i in tqdm(range(X.shape[0])):

        # Predict this example
        x = np.expand_dims(X[i], axis=0)
        y = np.squeeze(predict_instance(x, model, ohe_ct_sv, scaler_ct).T, axis=1)  # Predict example
        prediction = 1 if y[1] >= THRESHOLD else 0  # Model's classification
        predicted_class = CLASS_NAMES[prediction]
        row = [client_ids[i], n_weeks, predicted_class, y[1] * 100]

        # Explain this prediction
        if give_explanations:
            explanation = predict_and_explain(X[i], model, explainer, ohe_ct_sv, scaler_ct, NUM_FEATURES, NUM_SAMPLES)
            exp_tuples = explanation.as_list()
            for exp_tuple in exp_tuples:
                row.extend(list(exp_tuple))
        rows.append(row)

    # Convert results to a Pandas dataframe and save
    results_df = pd.DataFrame(rows, columns=col_names)
    if save_results:
        results_path = cfg['PATHS']['BULK_PREDICTIONS'] + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
        results_df.to_csv(results_df.to_csv(results_path, index_label=False, index=False))
    return results_df


def trending_prediction(data_path=None):
    '''
    Gets predictions and explanations from the provided raw data and adds a timestamp. Concatenates results with those
    of previous predictions on same client. Running this function periodically enables trend analysis for clients as the
    data changes over time.
    :param data_path: Path of raw data
    '''
    input_stream = open(os.getcwd() + "/config.yml", 'r')
    cfg = yaml.full_load(input_stream)
    trending_pred_path = cfg['PATHS']['TRENDING_PREDICTIONS']

    # Get model predictions and explanations.
    results_df = predict_and_explain_set(data_path=data_path, save_results=False)
    print(results_df.head())
    results_df.insert(0, 'Timestamp', pd.to_datetime(datetime.today()))     # Add a timestamp to these results

    # If previous prediction file exists, load it and append the predictions we just made.
    if os.path.exists(trending_pred_path):
        prev_results_df = pd.read_csv(trending_pred_path)
        results_df = pd.concat([prev_results_df, results_df], axis=0)

    # Save the updated trend analysis prediction spreadsheet
    results_df.to_csv(trending_pred_path, index_label=False, index=False)
    return


if __name__ == '__main__':
    results = predict_and_explain_set(data_path=None, save_results=True, give_explanations=True)
    # trending_prediction(data_path=None)