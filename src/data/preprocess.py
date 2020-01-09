import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yaml
import os

def load_df(path):
    '''
    Load a Pandas dataframe from a CSV file
    :param path: The file path of the CSV file
    :return: A Pandas dataframe
    '''
    # Read HIFIS data into a Pandas dataframe
    df = pd.read_csv(path, encoding="ISO-8859-1", low_memory=False)

    # Delete the first row of "-------" entries, as it is an artifact of the script used to pull down the database
    df.drop(df.index[0], inplace=True)
    return df

def cleanse_categorical_features(df, categorical_features):
    '''
    For all categorical features, replace entries containing unique values with "Other"
    :param df: a Pandas dataframe
    :param categorical_features: list of categorical features in the dataframe
    :return: the updated dataframe
    '''
    for feature in categorical_features:
        counts = df[feature].value_counts() # Get a Pandas series of all the values for
        single_values = counts[counts == 1] # Get values for this categorical feature unique to 1 row
        df[feature].loc[(df[feature].isin(list(single_values.index)))] = "Other" # Replace unique instances of categorical values with "Other"
    return df

def ohe_categorical_features(df, categorical_features):
    '''
    Converts categorical features to one-hot encoded format (i.e. vectorization) and appends to the dataframe
    :param df: A Pandas dataframe
    :param categorical_features: The names of the categorical features to encode
    :return: dataframe containing one-hot encoded features, list of one-hot encoded feature names
    '''
    ohe_categorical_features = []
    for feature in categorical_features:
        df_temp = pd.get_dummies(df[feature], prefix=feature)  # Create temporary dataframe of this feature one-hot encoded
        df = pd.concat((df, df_temp), axis=1)  # Concatenate temp one hot dataframe with original dataframe
        df = df.drop(feature, axis=1)  # Drop the original feature
        vectorized_headers_list = list(df_temp)
        for i in range(len(vectorized_headers_list)):
            ohe_categorical_features.append(vectorized_headers_list[i]) # Keep track of one-hot encoded features
    return df, ohe_categorical_features


def process_timestamps(df):
    '''
    Convert timestamps in raw date to datetimes
    :param df: A Pandas dataframe
    :return: The dataframe with its datetime fields updated accordingly
    '''
    features_list = list(df)  # Get a list of features
    for feature in features_list:
        if ('Date' in feature) or ('Start' in feature) or ('End' in feature):
            df[feature] = pd.to_datetime(df[feature], infer_datetime_format=True, errors='coerce')
    return df

def make_start_end_features(df, noncategorical_features, time_length_features):
    '''
    Create and add features for start and end times of certain features
    :param df: a Pandas dataframe
    :param noncategorical_features: list of noncategorical features in the dataset
    :param time_length_features: list of features that occur at events spaced in time
    :return: the updated dataframe, the updated list of noncategorical features
    '''
    for feature in time_length_features:
        feature_start_name = feature + 'StartDate'
        feature_end_name = feature + 'EndDate'
        df[feature_start_name] = np.where(df['ServiceType'] == feature, df['ServiceStartDate'], 0)
        df[feature_end_name] = np.where(df['ServiceType'] == feature, df['ServiceEndDate'], 0)
        noncategorical_features.extend([feature_start_name, feature_end_name])
    return df, noncategorical_features

def calculate_length_features(df, time_paired_features, time_length_features):
    '''
    Create features for total length of time for features with start and end dates
    :param df: a Pandas dataframe
    :param time_paired_features: Features that already had start and end dates in the original dataset
    :param time_length_features: Features whose start and end dates were calculated in this script
    :return:
    '''
    seconds_per_day = 60 * 60 * 24 # 60sec/min * 60min/hr * 24hr/day
    length_features = [] # Keep track of features identifying a time duration
    for service in time_paired_features:
        length_feature_name = 'LengthOf' + service['NAME'] + 'Days'
        df[length_feature_name] = (df[service['END']] - df[service['START']]).dt.total_seconds() / seconds_per_day
        length_features.append(length_feature_name)
    for feature in time_length_features:
        length_feature_name = 'LengthOf' + feature + 'Days'
        df[length_feature_name] = (df[feature + 'EndDate'] - df[feature + 'StartDate']).dt.total_seconds() / seconds_per_day
        length_features.append(length_feature_name)
    return df, length_features

def convert_yn_to_boolean(df, categorical_features, noncategorical_features):
    '''
    Convert yes/no features to boolean features. Avoids vectorization.
    :param df: a Pandas dataframe
    :return: updated dataframe with the yes/no features converted to boolean values
    '''
    for feature_name in categorical_features:
        if "YN" in feature_name:
            new_feature_name = feature_name[0:feature_name.index('YN')]
            df[new_feature_name] = np.where(df[feature_name] == 'Y', 1, 0)
            df.drop(feature_name, axis=1, inplace=True)
            categorical_features.remove(feature_name)
            noncategorical_features.append(new_feature_name)
    return df, categorical_features, noncategorical_features

def set_ground_truths(df, chronic_threshold, days, end_date):
    '''
    Determine ground truth for each client, which is defined as a certain number of days spent in a shelter
    :param df: a Pandas dataframe
    :param chronic_threshold: Minimum # of days spent in shelter to be considered chronically homeless
    :param days: Number of days over which to cound # days spent in shelter
    :param end_date: The last date of the time period to consider
    :return: the dataframe with a ground truth column appended at the end
    '''
    unique_clients = df['ClientID'].unique()  # Get a list of unique clients by ID
    print('Number of unique Client IDs in dataset: ', len(unique_clients))
    num_pos = num_neg = 0 # Keep track of number of clients in each class over the last year
    for client in unique_clients:
        client_stay_temp_mask = (df['ClientID'] == client) & (df['ServiceType'] == "Stay")  # Select rows depicting stays
        client_stay_df = df.loc[client_stay_temp_mask]
        client_stay_df = client_stay_df.sort_values(by=['ServiceStartDate'])

        # Create a ground truth based on if the chronic condition was met in the most recent period.
        start_date = end_date - timedelta(days=days)
        client_year_stays_df = client_stay_df.loc[start_date:end_date]
        total_days_spent = client_year_stays_df['LengthOfStayDays'].sum()  # Determine the length of stay in days
        single_client_mask = df['ClientID'] == client
        df.loc[single_client_mask, 'TotalDays'] = total_days_spent
        if total_days_spent >= chronic_threshold:
            df.loc[single_client_mask, 'GroundTruth'] = 1
            num_pos += 1
        else:
            df.loc[single_client_mask, 'GroundTruth'] = 0
            num_neg += 1
    return df, num_pos, num_neg

def condense_df(df, noncategorical_features, ohe_categorical_features):
    '''
    Build a dictionary of columns and arguments to feed into the grouping function, and group the dataframe
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
    for i in range(len(ohe_categorical_features)):
        temp_dict[ohe_categorical_features[i]] = 'max'  # Group one hot features by max value
    grouping_dictionary = {**grouping_dictionary, **temp_dict}
    temp_dict = {}
    for i in range(len(length_features)):
        temp_dict[length_features[i]] = 'sum'
    grouping_dictionary = {**grouping_dictionary, **temp_dict}
    temp_dict = {'IncomeTotal': 'first', 'FoodBankTrips': 'sum', 'GroundTruth': 'first', }
    grouping_dictionary = {**grouping_dictionary, **temp_dict}

    # Group the data by ClientID using the dictionary created above
    df_unique_clients = df.groupby(['ClientID']).agg(grouping_dictionary)
    return df_unique_clients


# Load config data
input_stream = open(os.getcwd() + "/config.yml", 'r')
config = yaml.full_load(input_stream)
categorical_features = config['DATA']['CATEGORICAL_FEATURES']
noncategorical_features = config['DATA']['NONCATEGORICAL_FEATURES']
GROUND_TRUTH_DURATION = 365     # In days. Set to 1 year.

# Load HIFIS database into Pandas dataframe
print("Loading HIFIS data.")
df = load_df(config['PATHS']['RAW_DATA'])

# Delete unwanted columns
print("Dropping some features.")
for feature in config['DATA']['FEATURES_TO_DROP_FIRST']:
    df.drop(feature, axis=1, inplace=True)

# Create a new feature for start and end of each service type that is accessed over multiple timeframes
print("Adding features for service start/end dates, family.")
df, noncategorical_features = make_start_end_features(df, noncategorical_features, config['DATA']['TIME_LENGTH_FEATURES'])

# Create a new boolean feature that indicates whether client has family
df['HasFamily'] = np.where(df['FamilyID'] != 'NULL', 1, 0)
noncategorical_features.append('HasFamily')

# Convert yes/no features to boolean features
print("Convert yes/no categorical features to boolean")
df, categorical_features, noncategorical_features = convert_yn_to_boolean(df, categorical_features, noncategorical_features)

# Convert all timestamps to datetime objects
print("Converting timestamps to datetimes.")
df = process_timestamps(df)

# Create length of service feature to describe the duration of timestamped features
print("Calculating length features.")
df, length_features = calculate_length_features(df, config['DATA']['TIME_PAIRED_FEATURES'], config['DATA']['TIME_LENGTH_FEATURES'])

# Add number of food bank trips as a feature
print("Add # visits to food bank as feature.")
df['FoodBankTrips'] = np.where((df['ServiceType'] == "Food Bank"), 1, 0)

# Index dataframe by the service start column
df = df.set_index('ServiceStartDate')

# Get total monthly incomes for each client and add as feature
print("Adding total monthly income as feature.")
df['MonthlyAmount'] = pd.to_numeric(df['MonthlyAmount'])
df['IncomeTotal'] = df.groupby(['ServiceID', 'ClientID', 'ServiceStartDate'])['MonthlyAmount'].transform('sum')

# Drop duplicate rows for ClientID and ServiceID. Should now have 1 row for each client.
df.drop_duplicates(subset=['ServiceID', 'ClientID'], keep='first', inplace=True)

# Compute ground truth for each client. Ground truth
print("Calculating ground truths.")
gt_end_date = pd.to_datetime(config['DATA']['GROUND_TRUTH_DATE'])
df, num_pos, num_neg = set_ground_truths(df, config['DATA']['CHRONIC_THRESHOLD'], GROUND_TRUTH_DURATION, gt_end_date)
# Log some useful stats
print("# clients in last year meeting homeslessness criteria = ", num_pos)
print("# clients in last year meeting homeslessness criteria = ", num_neg)
print("% positive for chronic homelessness = ", 100 * num_pos / (num_pos + num_neg))

# Amalgamate rows to have one entry per client
print("Grouping the dataframe.")
df_unique_clients = condense_df(df, noncategorical_features, categorical_features)

print("Cleansing categorical features of unique values.")
df_unique_clients = cleanse_categorical_features(df_unique_clients, categorical_features)

# One hot encode the categorical features
print("One-hot encoding categorical features.")
df_unique_clients, ohe_categorical_features = ohe_categorical_features(df_unique_clients, categorical_features)

# Drop unnecessary features
print("Dropping unnecessary features.")
for column in config['DATA']['FEATURES_TO_DROP_LAST']:
    df_unique_clients.drop(column, axis=1, inplace=True)

# Replace all instances of NaN in the dataframe with 0
df_unique_clients.fillna(0, inplace=True)

# Save vectorized data
print("Saving data.")
df_unique_clients.to_csv(config['PATHS']['VECTORIZED_DATA'], sep=',', header=True)


