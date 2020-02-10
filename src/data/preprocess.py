import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import yaml
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals.joblib import dump, load
from category_encoders import OrdinalEncoder

def load_df(path):
    '''
    Load a Pandas dataframe from a CSV file
    :param path: The file path of the CSV file
    :return: A Pandas dataframe
    '''
    # Read HIFIS data into a Pandas dataframe
    df = pd.read_csv(path, encoding="ISO-8859-1", low_memory=False)
    return df


def classify_cat_features(df, cat_features):
    '''
    Classify categorical features as either single- or multi-valued.
    :param df: Pandas dataframe
    :param cat_features: List of categorical features
    :return: list of single-valued categorical features, list of multi-valued categorical features
    '''
    def classify_features(client_df):
        '''
        Helper function for categorical feature classification, distributed across clients.
        :param client_df: Dataframe with 1 client's records
        '''
        for feature in cat_features:
            # If this feature takes more than 1 value per client, move it to the list of multi-valued features
            if client_df[feature].nunique() > 1:
                sv_cat_features.remove(feature)
                mv_cat_features.append(feature)
                return

    sv_cat_features = cat_features  # First, assume all categorical features are single-valued
    mv_cat_features = []
    df.groupby('ClientID').progress_apply(classify_features)
    return sv_cat_features, mv_cat_features

def vec_multi_value_cat_features(df, mv_cat_features, config, load_ct=False):
    '''
        Converts multi-valued categorical features to vectorized format and appends to the dataframe
        :param df: A Pandas dataframe
        :param mv_categorical_features: The names of the categorical features to vectorize
        :param config: project config
        :param load_ct: Flag indicating whether to load a saved column transformer
        :return: dataframe containing vectorized features, list of vectorized feature names
        '''
    orig_col_names = df.columns

    # One hot encode the multi-valued categorical features
    mv_cat_feature_idxs = [df.columns.get_loc(c) for c in mv_cat_features if c in df]  # List of categorical column indices
    if load_ct:
        col_trans_ohe = load(config['PATHS']['OHE_COL_TRANSFORMER_MV'])
        df_ohe = pd.DataFrame(col_trans_ohe.transform(df), index=df.index.copy())
    else:
        col_trans_ohe = ColumnTransformer(
            transformers=[('col_trans_mv_ohe', OneHotEncoder(sparse=False, handle_unknown='ignore', dtype=int), mv_cat_feature_idxs)],
            remainder='passthrough'
        )
        df_ohe = pd.DataFrame(col_trans_ohe.fit_transform(df), index=df.index.copy())
        dump(col_trans_ohe, config['PATHS']['OHE_COL_TRANSFORMER_MV'], compress=True)  # Save the column transformer

    # Build list of feature names for the new DataFrame
    mv_vec_cat_features = []
    for i in range(len(mv_cat_features)):
        feat_names = list(col_trans_ohe.transformers_[0][1].categories_[i])
        for j in range(len(feat_names)):
            mv_vec_cat_features.append(mv_cat_features[i] + '_' + feat_names[j])
    ohe_feat_names = mv_vec_cat_features.copy()
    for feat in orig_col_names:
        if feat not in mv_cat_features:
            ohe_feat_names.append(feat)
    df_ohe.columns = ohe_feat_names
    return df_ohe, mv_vec_cat_features

def vec_single_value_cat_features(df, sv_cat_features, config, load_ct=False):
    '''
    Converts single-valued categorical features to one-hot encoded format (i.e. vectorization) and appends to the dataframe.
    Keeps track of a mapping from feature indices to categorical values, for interpretability purposes.
    :param df: A Pandas dataframe
    :param sv_cat_features: The names of the categorical features to encode
    :param config: project config dict
    :param load_ct: Flag indicating whether to load saved column transformers
    :return: dataframe containing one-hot encoded features, list of one-hot encoded feature names
    '''
    # Convert single-valued categorical features to numeric data
    cat_feature_idxs = [df.columns.get_loc(c) for c in sv_cat_features if c in df]  # List of categorical column indices
    cat_value_names = {}  # Dictionary of categorical feature indices and corresponding names of feature values
    if load_ct:
        col_trans_ordinal = load(config['PATHS']['ORDINAL_COL_TRANSFORMER'])
        df[sv_cat_features] = col_trans_ordinal.transform(df) - 1
    else:
        col_trans_ordinal = ColumnTransformer(transformers=[('col_trans_ordinal', OrdinalEncoder(handle_unknown='value'), sv_cat_features)])
        df[sv_cat_features] = col_trans_ordinal.fit_transform(df) - 1   # Want integer representation of features to start at 0
        dump(col_trans_ordinal, config['PATHS']['ORDINAL_COL_TRANSFORMER'], compress=True)  # Save the column transformer

    # Preserve named values of each categorical feature
    for i in range(len(sv_cat_features)):
        cat_value_names[cat_feature_idxs[i]] = []
        for j in range(len(col_trans_ordinal.transformers_[0][1].category_mapping[i])):
            # Last one is nan; we don't want that
            cat_value_names[cat_feature_idxs[i]] = col_trans_ordinal.transformers_[0][1].category_mapping[i]['mapping'].index.tolist()[:-1]

    # One hot encode the single-valued categorical features
    if load_ct:
        col_trans_ohe = load(config['PATHS']['OHE_COL_TRANSFORMER_SV'])
        df_ohe = pd.DataFrame(col_trans_ohe.transform(df), index=df.index.copy())
    else:
        col_trans_ohe = ColumnTransformer(
            transformers=[('col_trans_ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'), cat_feature_idxs)],
            remainder='passthrough'
        )
        df_ohe = pd.DataFrame(col_trans_ohe.fit_transform(df), index=df.index.copy())
        dump(col_trans_ohe, config['PATHS']['OHE_COL_TRANSFORMER_SV'], compress=True)  # Save the column transformer

    # Build list of feature names for OHE dataset
    ohe_feat_names = []
    for i in range(len(sv_cat_features)):
        for value in cat_value_names[cat_feature_idxs[i]]:
            ohe_feat_names.append(sv_cat_features[i] + '_' + str(value))
    vec_sv_cat_features = ohe_feat_names.copy()
    for feat in df.columns:
        if feat not in sv_cat_features:
            ohe_feat_names.append(feat)
    df_ohe.columns = ohe_feat_names

    data_info = {}  # To store info for later use in LIME
    data_info['SV_CAT_FEATURES'] = sv_cat_features
    data_info['VEC_SV_CAT_FEATURES'] = vec_sv_cat_features
    data_info['SV_CAT_FEATURE_IDXS'] = cat_feature_idxs
    data_info['SV_CAT_VALUES'] = cat_value_names
    return df, df_ohe, data_info

def process_timestamps(df):
    '''
    Convert timestamps in raw date to datetimes
    :param df: A Pandas dataframe
    :return: The dataframe with its datetime fields updated accordingly
    '''
    features_list = list(df)  # Get a list of features
    for feature in features_list:
        if ('Date' in feature) or ('Start' in feature) or ('End' in feature) or (feature == 'DOB'):
            df[feature] = pd.to_datetime(df[feature], errors='coerce')
    return df

def remove_n_weeks(df, n_weeks, gt_end_date, dated_feats, cat_feats):
    '''
    Remove records from the dataframe that have timestamps in the n weeks leading up to the ground truth date
    :param df: Pandas dataframe
    :param n_weeks: number of recent weeks to remove records
    :param gt_end_date: the date used for ground truth calculation
    :return: updated dataframe with the relevant rows removed
    '''
    train_end_date = gt_end_date - timedelta(days=(n_weeks * 7))    # Maximum for training set records
    print("Train end date: ", train_end_date)
    df = df[df['ServiceStartDate'] <= train_end_date]               # Delete rows where service occurred after this date
    df['ServiceEndDate'] = df['ServiceEndDate'].clip(upper=train_end_date)  # Set end date for ongoing services to this date

    # Set features with dated events occurring after the maximum training set date to null
    for feat in dated_feats:
        idxs_to_update = df[df[feat] > train_end_date].index.tolist()
        dated_feats[feat] = [f for f in dated_feats[feat] if f in cat_feats]
        df.loc[idxs_to_update, dated_feats[feat]] = np.nan

    # Update client age
    if 'DOB' in df.columns:
        df['CurrentAge'] = (train_end_date - df['DOB']).astype('<m8[Y]')
    return df


def calculate_ground_truth(df, chronic_threshold, days, end_date):
    '''
    Iterate through dataset by client to calculate ground truth
    :param df: a Pandas dataframe
    :param chronic_threshold: Minimum # of days spent in shelter to be considered chronically homeless
    :param days: Number of days over which to cound # days spent in shelter
    :param end_date: The last date of the time period to consider
    :return: a DataSeries mapping ClientID to ground truth
    '''

    def client_gt(client_df):
        '''
        Helper function ground truth calculation.
        :param client_df: A dataframe containing all rows for a client
        :return: the client dataframe ground truth calculated correctly
        '''
        client_df.sort_values(by=['ServiceStartDate'], inplace=True) # Sort records by service start date
        gt_stays = 0 # Keep track of total stays, as well as # stays during ground truth time range
        last_stay_end = pd.to_datetime(0)
        last_stay_start = pd.to_datetime(0)

        # Iterate over all of client's records. Note itertuples() is faster than iterrows().
        for row in client_df.itertuples():
            stay_start = getattr(row, 'ServiceStartDate')
            stay_end = min(getattr(row, 'ServiceEndDate'), end_date) # If stay is ongoing through end_date, set end of stay as end_date
            if (stay_start > last_stay_start) and (stay_end > last_stay_end):
                if (stay_start.date() >= start_date.date()) or (stay_end.date() >= start_date.date()):
                    # Account for cases where stay start earlier than start of range, or stays overlapping from previous stay
                    stay_start = max(start_date, stay_start, last_stay_end)
                    if (stay_end - stay_start).total_seconds() >= min_stay_seconds:
                        gt_stays += (stay_end.date() - stay_start.date()).days + (stay_start.date() != last_stay_end.date())
                last_stay_end = stay_end
                last_stay_start = stay_start

        # Determine if client meets ground truth threshold
        if gt_stays >= chronic_threshold:
            client_df['GroundTruth'] = 1
        return client_df

    start_date = end_date - timedelta(days=days) # Get start of ground truth window
    min_stay_seconds = 60 * 15  # Stays must be at least 15 minutes
    df_temp = df[['ClientID', 'ServiceType', 'ServiceStartDate', 'ServiceEndDate']]
    df_temp['GroundTruth'] = 0
    df_temp = df_temp.loc[(df_temp['ServiceType'] == 'Stay')]
    df_temp = df_temp.groupby('ClientID').progress_apply(client_gt)
    ds_gt = df_temp['GroundTruth']
    ds_gt = ds_gt.groupby(['ClientID']).agg({'GroundTruth': 'max'})
    return ds_gt

def calculate_client_features(df, end_date, counted_services):
    '''
    Iterate through dataset by client to calculate some features (total stays, monthly income)
    :param df: a Pandas dataframe
    :param end_date: The last date of the time period to consider
    :return: the dataframe with the new features and ground truth appended at the end
    '''

    def client_features(client_df):
        '''
        Helper function for total stay, total income and ground truth calculation.
        To be used on a subset of the dataframe
        :param client_df: A dataframe containing all rows for a client
        :return: the client dataframe with total stays and ground truth columns appended
        '''
        client_df.sort_values(by=['ServiceStartDate'], inplace=True) # Sort records by service start date
        total_stays = 0 # Keep track of total stays, as well as # stays during ground truth time range
        last_stay_end = pd.to_datetime(0)   # Unix Epoch (1970-01-01 00:00:00)
        last_stay_start = pd.to_datetime(0)
        last_service_end = pd.to_datetime(0)
        last_service = ''

        # Iterate over all of client's records. Note itertuples() is faster than iterrows().
        for row in client_df.itertuples():
            service_start = getattr(row, 'ServiceStartDate')
            service_end = min(getattr(row, 'ServiceEndDate'), end_date) # If stay is ongoing through end_date, set end of stay as end_date
            if (getattr(row, 'ServiceType') == 'Stay'):
                if (service_start > last_stay_start) and (service_end > last_stay_end):
                    service_start = max(service_start, last_stay_end)   # Don't count any stays overlapping from previous stay
                    if (service_end - service_start).total_seconds() >= min_stay_seconds:
                        total_stays += (service_end.date() - service_start.date()).days + (service_start.date() != last_stay_end.date())
                    last_stay_end = service_end
                    last_stay_start = service_start
            elif (service_end != last_service_end) or (getattr(row, 'ServiceType') != last_service):
                service = getattr(row, 'ServiceType')
                client_df['Num_' + service] += 1    # Increment # of times this service was accessed by this client
                last_service_end = service_end
                last_service = service
        client_df['TotalStays'] = total_stays

        # Calculate total monthly income for client
        client_income_df = client_df.drop_duplicates(subset=['IncomeType'])
        client_df['IncomeTotal'] = client_income_df['MonthlyAmount'].sum()
        return client_df

    df['TotalStays'] = 0    # Create columns for stays and income total
    df['IncomeTotal'] = 0
    df['MonthlyAmount'] = pd.to_numeric(df['MonthlyAmount'])
    min_stay_seconds = 60 * 15  # Stays must be at least 15 minutes
    numerical_service_features = []
    for service in counted_services:
        df['Num_' + service] = 0
        numerical_service_features.append('Num_' + service)
    df_temp = df.copy()
    df_temp = df_temp.groupby('ClientID').progress_apply(client_features)
    df_temp = df_temp.droplevel('ClientID', axis='index')
    df.update(df_temp)  # Update all rows with corresponding stay length and total income
    return df, numerical_service_features

def aggregate_df(df, noncategorical_features, vec_mv_categorical_features, vec_sv_categorical_features, numerical_service_features):
    '''
    Build a dictionary of columns and arguments to feed into the aggregation function, and aggregate the dataframe
    :param df: a Pandas dataframe
    :param noncategorical_features: list of noncategorical features
    :param ohe_features: list of one-hot encoded features
    :return: A grouped dataframe with one row for each client
    '''
    grouping_dictionary = {}
    temp_dict = {}
    non_categorical_features = noncategorical_features
    non_categorical_features.remove('ServiceStartDate')
    non_categorical_features.remove('ClientID')

    # Create a dictionary of column names and function names to pass into the groupby function
    for i in range(len(non_categorical_features)):
        grouping_dictionary[non_categorical_features[i]] = 'first'  # Group noncategorical features by first occurrence
    for i in range(len(vec_sv_categorical_features)):
        temp_dict[vec_sv_categorical_features[i]] = 'first'  # Group single-valued categorical features by max value
    grouping_dictionary = {**grouping_dictionary, **temp_dict}
    temp_dict = {}
    for i in range(len(vec_mv_categorical_features)):
        temp_dict[vec_mv_categorical_features[i]] = lambda x: 1 if any(x) else 0  # Group multi-valued categorical features
    grouping_dictionary = {**grouping_dictionary, **temp_dict}
    temp_dict = {}
    for i in range(len(numerical_service_features)):
        temp_dict[numerical_service_features[i]] = 'first'  # Group one hot features by max value
    grouping_dictionary = {**grouping_dictionary, **temp_dict}
    temp_dict = {'IncomeTotal': 'first', 'TotalStays': 'max',}
    grouping_dictionary = {**grouping_dictionary, **temp_dict}
    if 'GroundTruth' in df.columns:
        temp_dict = {'GroundTruth': 'max', }
        grouping_dictionary = {**grouping_dictionary, **temp_dict}

    # Group the data by ClientID using the dictionary created above
    df_unique_clients = df.groupby(['ClientID']).agg(grouping_dictionary)
    return df_unique_clients


def preprocess(n_weeks=None, include_gt=True, calculate_gt=True, classify_cat_feats=True, load_ct=False, data_path=None):
    '''
    Load results of the HIFIS SQL query and process the data into features for model training or prediction.
    :param n_weeks: Prediction horizon [weeks]
    :param include_gt: Boolean describing whether to include ground truth in preprocessed data. Set False if using data to predict.
    :param calculate_gt: Boolean describing whether to load precomputed ground truth from disk. Set False to compute ground truth.
    :param classify_cat_feats: Boolean describing whether to classify categorical features as single- or multi-valued.
                               If False, load from disk.
    :param load_ct: Boolean describing whether to load pre-fitted data transformers
    :param data_path: Path to load raw data from
    :return: dict containing preprocessed data before and after encoding single-value categorical features
    '''
    run_start = datetime.today()
    tqdm.pandas()
    input_stream = open(os.getcwd() + "/config.yml", 'r')
    config = yaml.full_load(input_stream)       # Load config data
    categorical_features = config['DATA']['CATEGORICAL_FEATURES']
    noncategorical_features = config['DATA']['NONCATEGORICAL_FEATURES']
    features_to_drop_last = config['DATA']['FEATURES_TO_DROP_LAST']
    GROUND_TRUTH_DURATION = 365     # In days. Set to 1 year.
    if n_weeks is None:
        N_WEEKS = config['DATA']['N_WEEKS']
    else:
        N_WEEKS = n_weeks

    # Load HIFIS database into Pandas dataframe
    print("Loading HIFIS data.")
    if data_path == None:
        data_path = config['PATHS']['RAW_DATA']
    df = load_df(data_path)

    # Delete unwanted columns
    print("Dropping some features.")
    for feature in config['DATA']['FEATURES_TO_DROP_FIRST']:
        df.drop(feature, axis=1, inplace=True)
        if feature in noncategorical_features:
            noncategorical_features.remove(feature)
        elif feature in categorical_features:
            categorical_features.remove(feature)

    # Create a new boolean feature that indicates whether client has family
    df['HasFamily'] = np.where((~df['FamilyID'].isnull()), 'Y', 'N')
    categorical_features.append('HasFamily')
    noncategorical_features.remove('FamilyID')

    # Replace null ServiceEndDate entries with today's date. Assumes client is receiving ongoing services.
    df['ServiceEndDate'] = np.where(df['ServiceEndDate'].isnull(), pd.to_datetime('today'), df['ServiceEndDate'])

    # Convert all timestamps to datetime objects
    print("Converting timestamps to datetimes.")
    df = process_timestamps(df)

    # Calculate ground truth and save it. Or load pre-saved ground truth.
    gt_end_date = pd.to_datetime(config['DATA']['GROUND_TRUTH_DATE'])
    if include_gt:
        if calculate_gt:
            print("Calculating ground truth.")
            ds_gt = calculate_ground_truth(df, config['DATA']['CHRONIC_THRESHOLD'], GROUND_TRUTH_DURATION, gt_end_date)
            ds_gt.to_csv(config['PATHS']['GROUND_TRUTH'], sep=',', header=True)  # Save ground truth
        else:
            ds_gt = load_df(config['PATHS']['GROUND_TRUTH'])    # Load ground truth from file
            ds_gt = ds_gt.set_index('ClientID')
            ds_gt.index = ds_gt.index.astype(int)

    # Remove records from the database from n weeks ago and onwards
    print("Removing records ", N_WEEKS, " weeks back.")
    df = remove_n_weeks(df, N_WEEKS, gt_end_date, config['DATA']['OTHER_TIMED_FEATURES'], categorical_features)

    # Compute total stays, total monthly income, total # services accessed for each client.
    print("Calculating total stays, monthly income.")
    df, numerical_service_features = calculate_client_features(df, gt_end_date, config['DATA']['COUNTED_SERVICE_FEATURES'])
    noncategorical_features.extend(numerical_service_features)
    noncategorical_features.extend(['IncomeTotal', 'TotalStays'])

    # Index dataframe by the service start column
    df = df.set_index('ServiceStartDate')

    print("Separating multi and single-valued categorical features.")
    if classify_cat_feats:
        sv_cat_features, mv_cat_features = classify_cat_features(df, categorical_features)
    else:
        input_stream = open(os.getcwd() + config['PATHS']['DATA_INFO'], 'r')
        cfg_gen = yaml.full_load(input_stream)  # Get config data generated from previous preprocessing
        sv_cat_features = cfg_gen['SV_CAT_FEATURES']
        mv_cat_features = cfg_gen['MV_CAT_FEATURES']

    # Replace all instances of NaN in the dataframe with 0 or "Unknown"
    df[mv_cat_features] = df[mv_cat_features].fillna("None")

    # Vectorize the multi-valued categorical features
    print("Vectorizing multi-valued categorical features.")
    df, vec_mv_cat_features = vec_multi_value_cat_features(df, mv_cat_features, config, load_ct)

    # Amalgamate rows to have one entry per client
    print("Aggregating the dataframe.")
    df_clients = aggregate_df(df, noncategorical_features, vec_mv_cat_features, sv_cat_features, numerical_service_features)

    # Drop unnecessary features
    print("Dropping unnecessary features.")
    for column in features_to_drop_last:
        if column in df_clients.columns:
            df_clients.drop(column, axis=1, inplace=True)

    # Get list of remaining noncategorical features
    noncat_feats_gone = [f for f in noncategorical_features if f not in df_clients.columns]
    for feature in noncat_feats_gone:
        noncategorical_features.remove(feature)

    df_clients[sv_cat_features] = df_clients[sv_cat_features].fillna("Unknown")
    df_clients[noncategorical_features] = df_clients[noncategorical_features].fillna(0)

    # Vectorize single-valued categorical features. Keep track of feature names and values.
    print("Vectorizing single-valued categorical features.")
    df_clients, df_ohe_clients, data_info = vec_single_value_cat_features(df_clients, sv_cat_features, config, load_ct)

    # Append ground truth to dataset and log some useful stats about ground truth
    if include_gt:
        print("Appending ground truth.")
        df_clients.index = df_clients.index.astype(int)
        df_ohe_clients.index = df_ohe_clients.index.astype(int)
        df_clients = df_clients.join(ds_gt)  # Set ground truth for all clients to their saved values
        df_ohe_clients = df_ohe_clients.join(ds_gt)
        df_clients['GroundTruth'] = df_clients['GroundTruth'].fillna(0)
        df_ohe_clients['GroundTruth'] = df_ohe_clients['GroundTruth'].fillna(0)
        num_pos = df_clients['GroundTruth'].sum()  # Number of clients with positive ground truth
        num_neg = df_clients.shape[0] - num_pos  # Number of clients with negative ground truth
        print("# clients in last year meeting homelessness criteria = ", num_pos)
        print("# clients in last year not meeting homelessness criteria = ", num_neg)
        print("% positive for chronic homelessness = ", 100 * num_pos / (num_pos + num_neg))

    # If not preprocessing for prediction, save processed dataset
    print("Saving data.")
    if include_gt:
        df_clients.to_csv(config['PATHS']['PROCESSED_DATA'], sep=',', header=True)
        df_ohe_clients.to_csv(config['PATHS']['PROCESSED_OHE_DATA'], sep=',', header=True)

    # For producing interpretable results with categorical data:
    data_info['MV_CAT_FEATURES'] = mv_cat_features
    data_info['NON_CAT_FEATURES'] = noncategorical_features
    data_info['N_WEEKS'] = n_weeks
    with open(config['PATHS']['DATA_INFO'], 'w') as file:
        cat_feat_doc = yaml.dump(data_info, file)

    print("Runtime = ", ((datetime.today() - run_start).seconds / 60), " min")
    return df_clients

if __name__ == '__main__':
    preprocessed_data = preprocess(n_weeks=None, include_gt=True, calculate_gt=True, classify_cat_feats=True, load_ct=False)


