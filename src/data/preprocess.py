import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import yaml
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from joblib import dump, load
from category_encoders import OrdinalEncoder
from src.data.spdat import get_spdat_data

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
        :return List of single-valued categorical features, list of multi-valued categorical features
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


def get_mv_cat_feature_names(df, mv_cat_features):
    '''
    Build list of possible multi-valued categorical features
    :param df: DataFrame containing HIFIS data
    :param mv_cat_features: List of multi-valued categorical features
    :return: List of all individual multi-valued categorical features
    '''
    mv_vec_cat_features = []
    for f in mv_cat_features:
        mv_vec_cat_features += [(f + '_' + v) for v in list(df[f].unique()) if type(v) == str]
    return mv_vec_cat_features


def vec_multi_value_cat_features(df, mv_cat_features, cfg, load_ct=False, categories=None):
    '''
        Converts multi-valued categorical features to vectorized format and appends to the dataframe
        :param df: A Pandas dataframe
        :param mv_categorical_features: The names of the categorical features to vectorize
        :param cfg: project config
        :param load_ct: Flag indicating whether to load a saved column transformer
        :param categories: List of columns containing all possible values to encode
        :return: dataframe containing vectorized features, list of vectorized feature names
        '''
    orig_col_names = df.columns
    if categories is None:
        categories = 'auto'

    # One hot encode the multi-valued categorical features
    mv_cat_feature_idxs = [df.columns.get_loc(c) for c in mv_cat_features if c in df]  # List of categorical column indices
    if load_ct:
        col_trans_ohe = load(cfg['PATHS']['OHE_COL_TRANSFORMER_MV'])
        df_ohe = pd.DataFrame(col_trans_ohe.transform(df), index=df.index.copy())
    else:
        col_trans_ohe = ColumnTransformer(
            transformers=[('col_trans_mv_ohe', OneHotEncoder(categories=categories, sparse=False, handle_unknown='ignore', dtype=int), mv_cat_feature_idxs)],
            remainder='passthrough'
        )
        df_ohe = pd.DataFrame(col_trans_ohe.fit_transform(df), index=df.index.copy())
        dump(col_trans_ohe, cfg['PATHS']['OHE_COL_TRANSFORMER_MV'], compress=True)  # Save the column transformer

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

def vec_single_value_cat_features(df, sv_cat_features, cfg, load_ct=False):
    '''
    Converts single-valued categorical features to one-hot encoded format (i.e. vectorization) and appends to the dataframe.
    Keeps track of a mapping from feature indices to categorical values, for interpretability purposes.
    :param df: A Pandas dataframe
    :param sv_cat_features: The names of the categorical features to encode
    :param cfg: project config dict
    :param load_ct: Flag indicating whether to load saved column transformers
    :return: dataframe containing one-hot encoded features, list of one-hot encoded feature names
    '''
    # Convert single-valued categorical features to numeric data
    cat_feature_idxs = [df.columns.get_loc(c) for c in sv_cat_features if c in df]  # List of categorical column indices
    cat_value_names = {}  # Dictionary of categorical feature indices and corresponding names of feature values
    if load_ct:
        col_trans_ordinal = load(cfg['PATHS']['ORDINAL_COL_TRANSFORMER'])
        df[sv_cat_features] = col_trans_ordinal.transform(df)
    else:
        col_trans_ordinal = ColumnTransformer(transformers=[('col_trans_ordinal', OrdinalEncoder(handle_unknown='value'), sv_cat_features)])
        df[sv_cat_features] = col_trans_ordinal.fit_transform(df)   # Want integer representation of features to start at 0
        dump(col_trans_ordinal, cfg['PATHS']['ORDINAL_COL_TRANSFORMER'], compress=True)  # Save the column transformer

    # Preserve named values of each categorical feature
    for i in range(len(sv_cat_features)):
        cat_value_names[cat_feature_idxs[i]] = []
        for j in range(len(col_trans_ordinal.transformers_[0][1].category_mapping[i])):
            # Last one is nan; we don't want that
            cat_value_names[cat_feature_idxs[i]] = col_trans_ordinal.transformers_[0][1].category_mapping[i]['mapping'].index.tolist()[:-1]

    # One hot encode the single-valued categorical features
    if load_ct:
        col_trans_ohe = load(cfg['PATHS']['OHE_COL_TRANSFORMER_SV'])
        df_ohe = pd.DataFrame(col_trans_ohe.transform(df), index=df.index.copy())
    else:
        col_trans_ohe = ColumnTransformer(
            transformers=[('col_trans_ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'), cat_feature_idxs)],
            remainder='passthrough'
        )
        df_ohe = pd.DataFrame(col_trans_ohe.fit_transform(df), index=df.index.copy())
        dump(col_trans_ohe, cfg['PATHS']['OHE_COL_TRANSFORMER_SV'], compress=True)  # Save the column transformer

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

    cat_feat_info = {}  # To store info for later use in LIME
    cat_feat_info['SV_CAT_FEATURES'] = sv_cat_features
    cat_feat_info['VEC_SV_CAT_FEATURES'] = vec_sv_cat_features
    cat_feat_info['SV_CAT_FEATURE_IDXS'] = cat_feature_idxs

    # To use sparse matrices in LIME, ordinal encoded values must start at 1. Add dummy value to MV categorical features name lists.
    for i in range(len(sv_cat_features)):
        cat_value_names[cat_feature_idxs[i]].insert(0, 'DUMMY_VAL')
    cat_feat_info['SV_CAT_VALUES'] = cat_value_names
    return df, df_ohe, cat_feat_info

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

def remove_n_weeks(df, train_end_date, dated_feats):
    '''
    Remove records from the dataframe that have timestamps in the n weeks leading up to the ground truth date
    :param df: Pandas dataframe
    :param train_end_date: the most recent date that should appear in the dataset
    :param dated_feats: list of feature names with dated events
    :return: updated dataframe with the relevant rows removed
    '''
    df = df[df['ServiceStartDate'] <= train_end_date]               # Delete rows where service occurred after this date
    df['ServiceEndDate'] = df['ServiceEndDate'].clip(upper=train_end_date)  # Set end date for ongoing services to this date

    # Set features with dated events occurring after the maximum training set date to null
    for feat in dated_feats:
        idxs_to_update = df[df[feat] > train_end_date].index.tolist()
        dated_feats[feat] = [f for f in dated_feats[feat] if f in df.columns]
        df.loc[idxs_to_update, dated_feats[feat]] = np.nan

    # Update client age
    if 'DOB' in df.columns:
        df['CurrentAge'] = (train_end_date - df['DOB']).astype('<m8[Y]')
    return df.copy()


def calculate_ground_truth(df, chronic_threshold, days, end_date):
    '''
    Iterate through dataset by client to calculate ground truth
    :param df: a Pandas dataframe
    :param chronic_threshold: Minimum # of days spent in shelter to be considered chronically homeless
    :param days: Number of days over which to count # days spent in shelter
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
            service_type = getattr(row, 'ServiceType')
            if (stay_start > last_stay_start) and (stay_end > last_stay_end) and (service_type == 'Stay'):
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
    df_temp = df_temp.groupby('ClientID').progress_apply(client_gt)
    if df_temp.shape[0] == 0:
        return None
    if 'ClientID' not in df_temp.index:
        df_temp.set_index(['ClientID'], append=True, inplace=True)
    df_gt = df_temp['GroundTruth']
    df_gt = df_gt.groupby(['ClientID']).agg({'GroundTruth': 'max'})
    return df_gt


def calculate_client_features(df, end_date, noncat_feats, counted_services, timed_services, start_date=None):
    '''
    Iterate through dataset by client to calculate numerical features from services received by a client
    :param df: a Pandas dataframe
    :param end_date: The latest date of the time period to consider
    :param noncat_feats: List of noncategorical features
    :param counted_services: Service features for which we wish to count occurrences and create a feature for
    :param timed_services: Service features for which we wish to count total time received and create a feature for
    :param start_date: The earliest date of the time period to consider
    :return: the dataframe with the new service features included, updated list of noncategorical features
    '''

    def client_features(client_df):
        '''
        Helper function for total stay, total income and ground truth calculation.
        To be used on a subset of the dataframe
        :param client_df: A dataframe containing all rows for a client
        :return: the client dataframe with total stays and ground truth columns appended
        '''
        if start_date is not None:
            client_df = client_df[client_df['ServiceEndDate'] >= start_date]
            client_df['ServiceStartDate'].clip(lower=start_date, inplace=True)
        client_df = client_df[client_df['ServiceStartDate'] <= end_date]
        client_df['ServiceEndDate'].clip(upper=end_date, inplace=True) # If ongoing through end_date, set end as end_date
        client_df.sort_values(by=['ServiceStartDate'], inplace=True) # Sort records by service start date
        total_services = dict.fromkeys(total_timed_service_feats, 0) # Keep track of total days of service prior to training data end date
        last_service_end = dict.fromkeys(timed_services + counted_services, pd.to_datetime(0))   # Unix Epoch (1970-01-01 00:00:00)
        last_service_start = dict.fromkeys(timed_services + counted_services, pd.to_datetime(0))
        last_service = ''

        # Iterate over all of client's records. Note: itertuples() is faster than iterrows().
        for row in client_df.itertuples():
            service_start = getattr(row, 'ServiceStartDate')
            service_end = getattr(row, 'ServiceEndDate')
            service = getattr(row, 'ServiceType')
            if (service in timed_services):
                if (service_start > last_service_start[service]) and (service_end > last_service_end[service]):
                    service_start = max(service_start, last_service_end[service])   # Don't count any service overlapping from previous service
                    if (service == 'Stay') and ((service_end - service_start).total_seconds() < min_stay_seconds):
                        continue    # Don't count a stay if it's less than 15 minutes
                    total_services['Total_' + service] += (service_end.date() - service_start.date()).days + \
                                                               (service_start.date() != last_service_end[service].date())
                    last_service_end[service] = service_end
                    last_service_start[service] = service_start
            elif (service in counted_services) and \
                    ((service_end != last_service_end[service]) or (getattr(row, 'ServiceType') != last_service)):
                service = getattr(row, 'ServiceType')
                client_df['Num_' + service] += 1    # Increment # of times this service was accessed by this client
                last_service_end[service] = service_end
                last_service = service

        # Set total length of timed service features in client's records
        for feat in total_services:
            client_df[feat] = total_services[feat]

        # Calculate total monthly income for client
        client_income_df = client_df[['IncomeType', 'MonthlyAmount', 'IncomeStartDate', 'IncomeEndDate']]\
            .sort_values(by=['IncomeStartDate']).drop_duplicates(subset=['IncomeType'], keep='last')
        client_df['IncomeTotal'] = client_income_df['MonthlyAmount'].sum()
        return client_df

    total_timed_service_feats = ['Total_' + s for s in timed_services]
    for feat in total_timed_service_feats:
        df[feat] = 0
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

    # Update list of noncategorical features
    noncat_feats.extend(numerical_service_features)
    noncat_feats.extend(total_timed_service_feats)
    noncat_feats.extend(['IncomeTotal'])
    return df, noncat_feats


def calculate_ts_client_features(df, end_date, timed_services, counted_services, total_timed_service_feats,
                                      numerical_service_feats, feat_prefix, start_date=None):
    '''
    Iterate through dataset by client to calculate numerical features from services received by a client
    :param df: a Pandas dataframe
    :param end_date: The latest date of the time period to consider
    :param timed_services: Service features for which we wish to count total time received and create a feature for
    :param counted_services: Service features for which we wish to count occurrences and create a feature for
    :param total_timed_service_feats: Names of features to represent totals for timed service features
    :param numerical_service_feats: Names of features to represent totals for numerical service features
    :param feat_prefix: Prefix for total or timestep features
    :param start_date: The earliest date of the time period to consider
    :return: the dataframe with the new service features included, updated list of noncategorical features
    '''

    def client_services(client_df):
        '''
        Helper function for total stay, total income and ground truth calculation.
        To be used on a subset of the dataframe
        :param client_df: A dataframe containing all rows for a client
        :return: the client dataframe with total stays and ground truth columns appended
        '''
        if start_date is not None:
            client_df = client_df[client_df['ServiceEndDate'] >= start_date]
            client_df['ServiceStartDate'].clip(lower=start_date, inplace=True)
        client_df = client_df[client_df['ServiceStartDate'] <= end_date]
        client_df['ServiceEndDate'].clip(upper=end_date, inplace=True)
        client_df.sort_values(by=['ServiceStartDate', 'SPDAT_Date'], inplace=True) # Sort records by service start date
        total_services = dict.fromkeys(total_timed_service_feats, 0) # Keep track of total days of service prior to training data end date
        last_service_end = dict.fromkeys(timed_services + counted_services, pd.to_datetime(0))   # Unix Epoch (1970-01-01 00:00:00)
        last_service_start = dict.fromkeys(timed_services + counted_services, pd.to_datetime(0))
        last_service = ''

        # Iterate over all of client's records. Note: itertuples() is faster than iterrows().
        for row in client_df.itertuples():
            service_start = getattr(row, 'ServiceStartDate')
            service_end = getattr(row, 'ServiceEndDate')
            service = getattr(row, 'ServiceType')
            if (service in timed_services):
                if (service_start > last_service_start[service]) and (service_end > last_service_end[service]):
                    service_start = max(service_start, last_service_end[service])   # Don't count any service overlapping from previous service
                    if (service == 'Stay') and ((service_end - service_start).total_seconds() < min_stay_seconds):
                        continue    # Don't count a stay if it's less than 15 minutes
                    total_services[feat_prefix + service] += (service_end.date() - service_start.date()).days + \
                                                             (service_start.date() != last_service_end[service].date())
                    last_service_end[service] = service_end
                    last_service_start[service] = service_start
            elif (service in counted_services) and \
                    ((service_end != last_service_end[service]) or (getattr(row, 'ServiceType') != last_service)):
                service = getattr(row, 'ServiceType')
                client_df[feat_prefix + service] += 1    # Increment # of times this service was accessed by this client
                last_service_end[service] = service_end
                last_service = service

        # Set total length of timed service features in client's records
        for feat in total_services:
            client_df[feat] = total_services[feat]

        # Calculate total monthly income for client
        client_income_df = client_df[['IncomeType', 'MonthlyAmount', 'IncomeStartDate', 'IncomeEndDate']]\
            .sort_values(by=['IncomeStartDate']).drop_duplicates(subset=['IncomeType'], keep='last')
        client_df['IncomeTotal'] = client_income_df['MonthlyAmount'].sum()

        # If a client has multiple SPDAT records, ensure TotalScore is set to most recent value
        client_spdat_records = client_df[client_df['SPDAT_Date'] <= end_date]
        if client_spdat_records.shape[0] > 0:
            client_df['TotalScore'] = client_spdat_records['TotalScore'].iloc[-1]
        return client_df

    if df is None:
        return df
    df_temp = df.copy()
    min_stay_seconds = 60 * 15  # Stays must be at least 15 minutes
    end_date -= timedelta(seconds=1)    # To make calculations easier
    df_temp = df_temp.groupby('ClientID').progress_apply(client_services)
    if df_temp.shape[0] == 0:
        return None
    df_temp = df_temp.droplevel('ClientID', axis='index')
    return df_temp


def assemble_time_sequences(cfg, df_clients, noncat_feats):
    '''
    Appends most recent values for time series service features to data examples
    :param cfg: Project config
    :param df_clients: Dataframe of client data indexed by ClientID and Date
    :param noncat_feats: list of noncategorical features
    :return: Dataframe with recent time series service features, updated list of noncategorical features
    '''

    def client_windows(client_ts_df):
        '''
        Helper function to create examples with time series features going back T_X time steps for a client's records
        :param client_ts_df: A Dataframe of a client's time series service features
        :return: A Dataframe with client's current and past T_X time series service features in each row
        '''
        client_ts_df.sort_values(by=['Date'], ascending=False, inplace=True) # Sort records by date
        for i in range(1, T_X):
            for f in time_series_feats:
                client_ts_df['(-' + str(i) + ')' + f] = client_ts_df[f].shift(-i, axis=0)
        return client_ts_df

    T_X = cfg['DATA']['TIME_SERIES']['T_X']
    time_series_feats = [f for f in df_clients.columns if '-Day_' in f]
    df_ts_idx = list(df_clients.columns).index(time_series_feats[0])    # Get column number of first time series feature
    for i in range(1, T_X):
        for f in reversed(time_series_feats):
            new_ts_feat = '(-' + str(i) + ')' + f
            df_clients.insert(df_ts_idx, new_ts_feat, 0)
            noncat_feats.append(new_ts_feat)
    df_clients = df_clients.groupby('ClientID', group_keys=False).progress_apply(client_windows)

    # Records at the beginning of a client's experience should have 0 for past time series feats
    df_clients.fillna(0, inplace=True)

    # Cut off any trailing records that could have possible false 0's
    N_WEEKS = cfg['DATA']['N_WEEKS']
    DAYS_PER_YEAR = 365.25
    cutoff_date = pd.to_datetime(cfg['DATA']['GROUND_TRUTH_DATE']) - timedelta(days=N_WEEKS * 7) - \
                                      timedelta(days=int(cfg['DATA']['TIME_SERIES']['YEARS_OF_DATA'] * DAYS_PER_YEAR))
    df_clients = df_clients[df_clients.index.get_level_values(1) >= cutoff_date]
    return df_clients, noncat_feats


def aggregate_df(df, noncat_feats, vec_mv_cat_feats, vec_sv_cat_feats):
    '''
    Build a dictionary of columns and arguments to feed into the aggregation function, and aggregate the dataframe
    :param df: a Pandas dataframe
    :param noncat_feats: list of noncategorical features
    :param vec_mv_cat_feats: list of one-hot encoded multi-valued categorical features
    :param vec_sv_cat_feats: list of one-hot encoded single-valued categorical features
    :return: A grouped dataframe with one row for each client
    '''
    grouping_dictionary = {}
    temp_dict = {}
    if 'ServiceStartDate' in noncat_feats:
        noncat_feats.remove('ServiceStartDate')
    if 'ClientID' in noncat_feats:
        noncat_feats.remove('ClientID')

    # Create a dictionary of column names and function names to pass into the groupby function
    for i in range(len(noncat_feats)):
        if noncat_feats[i] in df.columns:
            grouping_dictionary[noncat_feats[i]] = 'max'  # Group noncategorical features by max value
    for i in range(len(vec_sv_cat_feats)):
        if vec_sv_cat_feats[i] in df.columns:
            temp_dict[vec_sv_cat_feats[i]] = 'first'  # Group single-valued categorical features by first occurrence
    grouping_dictionary = {**grouping_dictionary, **temp_dict}
    temp_dict = {}
    for i in range(len(vec_mv_cat_feats)):
        temp_dict[vec_mv_cat_feats[i]] = lambda x: 1 if any(x) else 0  # Group multi-valued categorical features by presence
    grouping_dictionary = {**grouping_dictionary, **temp_dict}
    if 'GroundTruth' in df.columns:
        temp_dict = {'GroundTruth': 'max', }
        grouping_dictionary = {**grouping_dictionary, **temp_dict}

    # Group the data by ClientID (and Date if time series) using the dictionary created above
    groupby_feats = ['ClientID']
    if 'Date' in df.columns:
        groupby_feats += ['Date']
    df_unique_clients = df.groupby(groupby_feats).agg(grouping_dictionary)
    return df_unique_clients


def calculate_gt_and_service_feats(cfg, df, categorical_feats, noncategorical_feats, gt_duration, include_gt, calculate_gt):
    # Calculate ground truth and save it. Or load pre-saved ground truth.
    n_weeks = cfg['DATA']['N_WEEKS']
    timed_service_feats = cfg['DATA']['TIMED_SERVICE_FEATURES']
    counted_service_feats = cfg['DATA']['COUNTED_SERVICE_FEATURES']
    gt_end_date = pd.to_datetime(cfg['DATA']['GROUND_TRUTH_DATE'])

    train_end_date = gt_end_date - timedelta(days=(n_weeks * 7))    # Maximum for training set records
    if include_gt:
        if calculate_gt:
            print("Calculating ground truth.")
            ds_gt = calculate_ground_truth(df, cfg['DATA']['CHRONIC_THRESHOLD'], gt_duration, gt_end_date)
            ds_gt.to_csv(cfg['PATHS']['GROUND_TRUTH'], sep=',', header=True)  # Save ground truth
        else:
            ds_gt = load_df(cfg['PATHS']['GROUND_TRUTH'])    # Load ground truth from file
            ds_gt = ds_gt.set_index('ClientID')
            ds_gt.index = ds_gt.index.astype(int)

    # Remove records from the database from n weeks ago and onwards
    print("Removing records ", n_weeks, " weeks back. Cutting off at ", train_end_date)
    df = remove_n_weeks(df, train_end_date, cfg['DATA']['TIMED_EVENT_FEATURES'])

    # Compute total stays, total monthly income, total # services accessed for each client.
    print("Calculating total service features, monthly income total.")
    df, noncategorical_feats = calculate_client_features(df, train_end_date, noncategorical_feats,
                                                         counted_service_feats, timed_service_feats)
    return df, ds_gt, noncategorical_feats


def calculate_time_series(cfg, cat_feat_info, df, categorical_feats, noncategorical_feats, gt_duration, include_gt, calculate_gt):
    '''
    Calculates ground truth, service time series features for client, client monthly income. Vectorizes multi-valued
    categorical features. Aggregates data to be indexed by ClientID and Date.
    :param cfg: Project config dict
    :param cat_feat_info: Generated config with feature information
    :param df: Dataframe of raw data
    :param categorical_feats: List of categorical features
    :param noncategorical_feats: List of noncategorical features
    :param gt_duration: Length of time used to calculate chronic homelessness ground truth
    :param include_gt: Boolean indicating whether to include ground truth in processed data
    :param calculate_gt: Boolean indicating whether to calculate ground truth
    :return: Aggreated client Dataframe with time series features and one-hot encoded multi-valued categorical features,
             Dataframe containing client ground truth, updated list of noncategorical features, list of all multi-valued
             categorical variables
    '''

    gt_end_date = pd.to_datetime(cfg['DATA']['GROUND_TRUTH_DATE'])
    n_weeks = cfg['DATA']['N_WEEKS']
    timed_service_feats = cfg['DATA']['TIMED_SERVICE_FEATURES']
    counted_service_feats = cfg['DATA']['COUNTED_SERVICE_FEATURES']
    TIME_STEP = cfg['DATA']['TIME_SERIES']['TIME_STEP']             # Size of timestep (in days)
    T_X = cfg['DATA']['TIME_SERIES']['T_X']                         # length of input sequence (in timesteps)
    DAYS_PER_YEAR = 365.25
    EARLIEST_GT_DATE = pd.to_datetime(cfg['DATA']['GROUND_TRUTH_DATE']) - timedelta(days=T_X * TIME_STEP) - \
                                      timedelta(days=int(cfg['DATA']['TIME_SERIES']['YEARS_OF_DATA'] * DAYS_PER_YEAR))
    print("Earliest time series ground truth date: ", EARLIEST_GT_DATE)     # Corresponds to earliest record for client stay

    if not include_gt:
        num_iterations = T_X
    else:
        num_iterations = (gt_end_date - EARLIEST_GT_DATE).days // TIME_STEP
    print('# of time series iterations:', num_iterations)

    sv_cat_feats = cat_feat_info['SV_CAT_FEATURES']
    mv_cat_feats = cat_feat_info['MV_CAT_FEATURES']
    if include_gt:
        all_mv_cat_feats = get_mv_cat_feature_names(df, mv_cat_feats)
    else:
        all_mv_cat_feats = cat_feat_info['VEC_MV_CAT_FEATURES']

    df_gt = pd.DataFrame()
    df_clients_time_series = pd.DataFrame()

    df['MonthlyAmount'] = pd.to_numeric(df['MonthlyAmount'])
    if 'IncomeTotal' not in noncategorical_feats:
        noncategorical_feats.append('IncomeTotal')
    df['IncomeTotal'] = 0

    timestep_prefix = str(TIME_STEP) + '-Day_'
    total_prefix = 'Total_'
    ts_timed_service_feats = [timestep_prefix + s for s in timed_service_feats]
    total_timed_service_feats = [total_prefix + s for s in timed_service_feats]
    ts_numerical_service_feats = [timestep_prefix + s for s in counted_service_feats]
    total_numerical_service_feats = [total_prefix + s for s in counted_service_feats]
    for f in ts_timed_service_feats + ts_numerical_service_feats + total_timed_service_feats + \
             total_numerical_service_feats:
        df[f] = 0
        if f not in noncategorical_feats:
            noncategorical_feats.append(f)
    for i in range(num_iterations):
        end_date = gt_end_date - timedelta(days=(TIME_STEP * i))

        # Go back in time (TIME_STEP * i / 7) weeks
        df_temp = remove_n_weeks(df, end_date, cfg['DATA']['TIMED_EVENT_FEATURES'])

        train_end_date = end_date - timedelta(days=(n_weeks * 7))  # Maximum for training set records

        # Calculate ground truth
        if include_gt:
            if calculate_gt:
                print('Calculating ground truth at ' + str(end_date))
                df_gt_cur = calculate_ground_truth(df_temp, cfg['DATA']['CHRONIC_THRESHOLD'], gt_duration,
                                                    end_date)
                if df_gt_cur is None:
                    continue
                df_gt_cur.insert(0, 'Date', train_end_date)     # Insert Date column with train end date as index
                df_gt = pd.concat([df_gt, df_gt_cur], axis=0, sort=False)

            # Remove records from the database from n weeks ago and onwards
            df_temp = remove_n_weeks(df_temp, train_end_date, cfg['DATA']['TIMED_EVENT_FEATURES'])
            cutoff_date = train_end_date
        else:
            cutoff_date = end_date

        # Compute weekly and total stays + services accessed for each client for each timestep.
        print('Calculating client service features at ' + str(train_end_date))
        start_date = cutoff_date - timedelta(days=TIME_STEP)
        df_temp = calculate_ts_client_features(df_temp, cutoff_date, timed_service_feats,
                                               counted_service_feats, total_timed_service_feats,
                                               total_numerical_service_feats, total_prefix,
                                               start_date=None)
        if df_temp is None:
            continue
        df_temp_timestep = calculate_ts_client_features(df_temp, cutoff_date, timed_service_feats,
                                                        counted_service_feats, ts_timed_service_feats,
                                                        ts_numerical_service_feats, timestep_prefix,
                                                        start_date=start_date)
        df_temp.update(df_temp_timestep)
        df_temp[ts_timed_service_feats + ts_numerical_service_feats + total_timed_service_feats + total_numerical_service_feats]\
            .fillna(0, inplace=True)
        df_temp = df_temp.set_index('ServiceStartDate')

        # Encode multi-valued categorical variables and aggregate the DataFrame
        df_temp[mv_cat_feats] = df_temp[mv_cat_feats].fillna("None")
        for f in all_mv_cat_feats:
            df_temp[f] = 0
        df_temp_ohe, vec_mv_cat_feats = vec_multi_value_cat_features(df_temp, mv_cat_feats, cfg, False)
        for f in vec_mv_cat_feats:
            df_temp[f] = df_temp_ohe[f]
        df_temp.insert(0, 'Date', cutoff_date)  # Insert Date column with train end date as index
        df_clients = aggregate_df(df_temp, noncategorical_feats, all_mv_cat_feats, sv_cat_feats)
        df_clients_time_series = pd.concat([df_clients_time_series, df_clients], axis=0, sort=False)
        df_clients_time_series[all_mv_cat_feats] = df_clients_time_series[all_mv_cat_feats].fillna(0)

    if include_gt:
        if calculate_gt:
            df_gt.set_index(['Date'], append=True, inplace=True)  # Index by date in addition to ClientID
            df_gt.to_csv(cfg['PATHS']['GROUND_TRUTH'], sep=',', header=True)  # Save ground truth
        else:
            df_gt = load_df(cfg['PATHS']['GROUND_TRUTH'])  # Load ground truth from file
            df_gt = df_gt.set_index('ClientID')
            df_gt.index = df_gt.index.astype(int)
    return df_clients_time_series, df_gt, noncategorical_feats, all_mv_cat_feats


def preprocess(cfg=None, n_weeks=None, include_gt=True, calculate_gt=True, classify_cat_feats=True, load_ct=False, data_path=None):
    '''
    Load results of the HIFIS SQL query and process the data into features for model training or prediction.
    :param cfg: Custom config object
    :param n_weeks: Prediction horizon [weeks]
    :param include_gt: Boolean describing whether to include ground truth in preprocessed data. Set False if using data to predict.
    :param calculate_gt: Boolean describing whether to compute ground truth or load from disk. Set True to compute ground truth.
    :param classify_cat_feats: Boolean describing whether to classify categorical features as single- or multi-valued.
                               If False, load from disk.
    :param load_ct: Boolean describing whether to load pre-fitted data transformers
    :param data_path: Path to load raw data from
    :return: dict containing preprocessed data before and after encoding single-value categorical features
    '''
    run_start = datetime.today()
    tqdm.pandas()
    if cfg is None:
        cfg = yaml.full_load(open("./config.yml", 'r'))       # Load project config data

    # Load lists of features in raw data
    categorical_feats = cfg['DATA']['CATEGORICAL_FEATURES']
    noncategorical_feats = cfg['DATA']['NONCATEGORICAL_FEATURES']
    identifying_feats_to_drop_last = cfg['DATA']['IDENTIFYING_FEATURES_TO_DROP_LAST']
    timed_feats_to_drop_last = cfg['DATA']['TIMED_FEATURES_TO_DROP_LAST']
    GROUND_TRUTH_DURATION = 365     # In days. Set to 1 year.

    # Set prediction horizon
    if n_weeks is None:
        N_WEEKS = cfg['DATA']['N_WEEKS']
    else:
        N_WEEKS = n_weeks
        cfg['DATA']['N_WEEKS'] = N_WEEKS

    # Load HIFIS database into Pandas dataframe
    print("Loading HIFIS data.")
    if data_path == None:
        data_path = cfg['PATHS']['RAW_DATA']
    df = load_df(data_path)

    # Exclude clients who did not provide consent to use their information for this project
    df.drop(df[df['ClientID'].isin(cfg['DATA']['CLIENT_EXCLUSIONS'])].index, inplace=True)

    # Delete unwanted columns
    print("Dropping some features.")
    for feature in cfg['DATA']['FEATURES_TO_DROP_FIRST']:
        df.drop(feature, axis=1, inplace=True)
        if feature in noncategorical_feats:
            noncategorical_feats.remove(feature)
        elif feature in categorical_feats:
            categorical_feats.remove(feature)

    # Create a new boolean feature that indicates whether client has family
    df['HasFamily'] = np.where((~df['FamilyID'].isnull()), 'Y', 'N')
    categorical_feats.append('HasFamily')
    noncategorical_feats.remove('FamilyID')

    # Replace null ServiceEndDate entries with today's date. Assumes client is receiving ongoing services.
    df['ServiceEndDate'] = np.where(df['ServiceEndDate'].isnull(), pd.to_datetime('today'), df['ServiceEndDate'])

    # Convert all timestamps to datetime objects
    print("Converting timestamps to datetimes.")
    df = process_timestamps(df)

    if cfg['TRAIN']['MODEL_DEF'] == 'hifis_rnn_mlp':
        print("Separating multi and single-valued categorical features.")
        if classify_cat_feats:
            sv_cat_feats, mv_cat_feats = classify_cat_features(df, categorical_feats)
            cat_feat_info = {}
            cat_feat_info['MV_CAT_FEATURES'] = mv_cat_feats
            cat_feat_info['SV_CAT_FEATURES'] = sv_cat_feats
        else:
            cat_feat_info = yaml.full_load(open(cfg['PATHS']['DATA_INFO'], 'r'))  # Get config data generated from previous preprocessing
            sv_cat_feats = cat_feat_info['SV_CAT_FEATURES']
            mv_cat_feats = cat_feat_info['MV_CAT_FEATURES']
            noncategorical_feats = cat_feat_info['NON_CAT_FEATURES']

        # Compute weekly and total service usage time series data for each client.
        print("Calculating cumulative and time series service features.")
        df_clients, df_gt, noncategorical_feats, all_mv_cat_feats = calculate_time_series(cfg, cat_feat_info, df, categorical_feats,
                                                                        noncategorical_feats, GROUND_TRUTH_DURATION,
                                                                        include_gt, calculate_gt)
    else:
        # Compute total stays, total monthly income, total # services accessed for each client.
        print("Calculating total service features, monthly income total.")
        df, df_gt, noncategorical_feats = calculate_gt_and_service_feats(cfg, df, categorical_feats, noncategorical_feats,
                                                                          GROUND_TRUTH_DURATION, include_gt, calculate_gt)
        categorical_feats.remove('ServiceType')

        # Index dataframe by the service start column
        df = df.set_index('ServiceStartDate')

        print("Separating multi and single-valued categorical features.")
        if classify_cat_feats:
            sv_cat_feats, mv_cat_feats = classify_cat_features(df, categorical_feats)
            all_mv_cat_feats = get_mv_cat_feature_names(df, mv_cat_feats)
        else:
            cat_feat_info = yaml.full_load(open(cfg['PATHS']['DATA_INFO'], 'r'))  # Get config data generated from previous preprocessing
            sv_cat_feats = cat_feat_info['SV_CAT_FEATURES']
            mv_cat_feats = cat_feat_info['MV_CAT_FEATURES']
            all_mv_cat_feats = cat_feat_info['VEC_MV_CAT_FEATURES']
            noncategorical_feats = cat_feat_info['NON_CAT_FEATURES']

        # Replace all instances of NaN in the dataframe with 0 or "Unknown"
        df[mv_cat_feats] = df[mv_cat_feats].fillna("None")

        # Vectorize the multi-valued categorical features
        print("Vectorizing multi-valued categorical features.")
        df, vec_mv_cat_feats = vec_multi_value_cat_features(df, mv_cat_feats, cfg, load_ct)

        # Amalgamate rows to have one entry per client
        print("Aggregating the dataframe.")
        df_clients = aggregate_df(df, noncategorical_feats, vec_mv_cat_feats, sv_cat_feats)

    # Include SPDAT data
    if cfg['DATA']['SPDAT']['INCLUDE_SPDATS'] and cfg['TRAIN']['MODEL_DEF'] != 'hifis_rnn_mlp':
        print("Adding SPDAT questions as features.")
        train_end_date = pd.to_datetime(cfg['DATA']['GROUND_TRUTH_DATE']) - timedelta(days=(n_weeks * 7))
        spdat_df, sv_cat_spdat_feats, noncat_spdat_feats = get_spdat_data(cfg['PATHS']['RAW_SPDAT_DATA'],
                                                                          train_end_date)
        if cfg['DATA']['SPDAT']['SPDAT_CLIENTS_ONLY']:
            df_clients = df_clients.join(spdat_df, how='inner')      # Add SPDAT data, but only take clients with SPDATs
            if cfg['DATA']['SPDAT']['SPDAT_DATA_ONLY']:
                df_clients = df_clients[list(spdat_df.columns)]  # Only features will be SPDAT questions
        else:
            df_clients = df_clients.join(spdat_df, how='left')      # Add SPDAT data for clients with SPDATs
        if classify_cat_feats:
            noncategorical_feats += noncat_spdat_feats
            sv_cat_feats += sv_cat_spdat_feats

    # Drop unnecessary features
    print("Dropping unnecessary features.")
    features_to_drop_last = identifying_feats_to_drop_last + timed_feats_to_drop_last
    for column in features_to_drop_last:
        if column in df_clients.columns:
            df_clients.drop(column, axis=1, inplace=True)

    # Get lists of remaining features
    noncat_feats_gone = [f for f in noncategorical_feats if f not in df_clients.columns]
    for feature in noncat_feats_gone:
        noncategorical_feats.remove(feature)
    sv_feats_gone = [f for f in sv_cat_feats if f not in df_clients.columns]
    for feature in sv_feats_gone:
        sv_cat_feats.remove(feature)

    # Fill nan values
    df_clients[sv_cat_feats] = df_clients[sv_cat_feats].fillna("Unknown")
    df_clients[noncategorical_feats] = df_clients[noncategorical_feats].fillna(-1)

    # Create columns for most recent T_X values of time series service features and place at the end of the DataFrame
    if cfg['TRAIN']['MODEL_DEF'] == 'hifis_rnn_mlp':
        print("Creating columns for recent past values of time series features.")
        df_clients, noncategorical_feats = assemble_time_sequences(cfg, df_clients, noncategorical_feats)
        new_col_order = list(df_clients.columns).copy()
        for col_name in list(df_clients.columns):
            if '-Day_' in col_name:
                new_col_order.append(new_col_order.pop(new_col_order.index(col_name)))
        df_clients = df_clients.reindex(columns=new_col_order)

    # Vectorize single-valued categorical features. Keep track of feature names and values.
    print("Vectorizing single-valued categorical features.")
    df_clients, df_ohe_clients, cat_feat_info = vec_single_value_cat_features(df_clients, sv_cat_feats, cfg, load_ct)

    # Append ground truth to dataset and log some useful stats about ground truth
    if include_gt:
        if cfg['TRAIN']['MODEL_DEF'] == 'hifis_rnn_mlp':
            df_clients = df_clients.join(df_gt)  # Set ground truth for all clients to their saved values
            df_ohe_clients = df_ohe_clients.join(df_gt)  # Set ground truth for all clients to their saved values
            df_clients['GroundTruth'] = df_clients['GroundTruth'].fillna(0)
            df_ohe_clients['GroundTruth'] = df_ohe_clients['GroundTruth'].fillna(0)
            num_pos = df_clients['GroundTruth'].sum()  # Number of clients with positive ground truth
            num_neg = df_clients.shape[0] - num_pos  # Number of clients with negative ground truth
            print("# time series client records meeting homelessness criteria = ", num_pos)
            print("# time series client records NOT meeting homelessness criteria = ", num_neg)
            print("% time series records positive for chronic homelessness = ", 100 * num_pos / (num_pos + num_neg))
        else:
            print("Appending ground truth.")
            df_clients.index = df_clients.index.astype(int)
            df_ohe_clients.index = df_ohe_clients.index.astype(int)
            df_clients = df_clients.join(df_gt)  # Set ground truth for all clients to their saved values
            df_ohe_clients = df_ohe_clients.join(df_gt)
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
        df_clients.to_csv(cfg['PATHS']['PROCESSED_DATA'], sep=',', header=True)
        df_ohe_clients.to_csv(cfg['PATHS']['PROCESSED_OHE_DATA'], sep=',', header=True)

    # For producing interpretable results with categorical data:
    cat_feat_info['MV_CAT_FEATURES'] = mv_cat_feats
    cat_feat_info['NON_CAT_FEATURES'] = noncategorical_feats
    cat_feat_info['VEC_MV_CAT_FEATURES'] = all_mv_cat_feats
    if include_gt:
        cat_feat_info['N_WEEKS'] = N_WEEKS      # Save the predictive horizon if we aren't preprocessing for prediction
    else:
        old_cat_feat_info = yaml.full_load(open(cfg['PATHS']['DATA_INFO'], 'r'))
        cat_feat_info['N_WEEKS'] = old_cat_feat_info['N_WEEKS']     # Get predictive horizon from previous preprocessing records
    with open(cfg['PATHS']['DATA_INFO'], 'w') as file:
        cat_feat_doc = yaml.dump(cat_feat_info, file)

    print("Runtime = ", ((datetime.today() - run_start).seconds / 60), " min")
    return df_clients

if __name__ == '__main__':
    preprocessed_data = preprocess(cfg=None, n_weeks=None, include_gt=True, calculate_gt=True,
                                   classify_cat_feats=True, load_ct=False)