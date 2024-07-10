import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from joblib import dump
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorboard.plugins.hparams import api as hp
from src.models.models import *
from src.custom.metrics import F1Score
from src.visualization.visualize import *

def get_class_weights(num_pos, num_neg, pos_weight=0.5):
    '''
    Computes weights for each class to be applied in the loss function during training.
    :param num_pos: # positive samples
    :param num_neg: # negative samples
    :return: A dictionary containing weights for each class
    '''
    weight_neg = (1 - pos_weight) * (num_neg + num_pos) / (num_neg)
    weight_pos = pos_weight * (num_neg + num_pos) / (num_pos)
    class_weight = {0: weight_neg, 1: weight_pos}
    print("Class weights: Class 0 = {:.2f}, Class 1 = {:.2f}".format(weight_neg, weight_pos))
    return class_weight

def minority_oversample(X_train, Y_train, algorithm='random_oversample'):
    '''
    Oversample the minority class using the specified algorithm
    :param X_train: Training set features
    :param Y_train: Training set labels
    :param algorithm: The oversampling algorithm to use. One of {"random_oversample", "smote", "adasyn"}
    :return: A new training set containing oversampled examples
    '''
    if algorithm == 'random_oversample':
        sampler = RandomOverSampler(random_state=np.random.randint(0, high=1000))
    elif algorithm == 'smote':
        sampler = SMOTE(random_state=np.random.randint(0, high=1000))
    elif algorithm == 'adasyn':
        sampler = ADASYN(random_state=np.random.randint(0, high=1000))
    else:
        sampler = RandomOverSampler(random_state=np.random.randint(0, high=1000))
    X_resampled, Y_resampled = sampler.fit_resample(X_train, Y_train)
    print("Train set shape before oversampling: ", X_train.shape, " Train set shape after resampling: ", X_resampled.shape)
    return X_resampled, Y_resampled


def load_dataset(cfg):
    '''
    Load the dataset from disk and partition into train/val/test sets. Normalize numerical data.
    :param cfg: Project config (from config.yml)
    :return: A dict of partitioned and normalized datasets, split into examples and labels
    '''

    # Load data info generated during preprocessing
    data = {}
    data['METADATA'] = {}
    input_stream = open(cfg['PATHS']['DATA_INFO'], 'r')
    data_info = yaml.full_load(input_stream)
    data['METADATA']['N_WEEKS'] = data_info['N_WEEKS']
    noncat_features = data_info['NON_CAT_FEATURES']   # Noncategorical features to be scaled

    # Load and partition dataset
    df_ohe = pd.read_csv(cfg['PATHS']['PROCESSED_OHE_DATA'])
    df = pd.read_csv(cfg['PATHS']['PROCESSED_DATA'])   # Data prior to one hot encoding
    train_split = cfg['TRAIN']['TRAIN_SPLIT']
    val_split = cfg['TRAIN']['VAL_SPLIT']
    test_split = cfg['TRAIN']['TEST_SPLIT']
    random_state = np.random.randint(0, high=1000)
    train_df_ohe, test_df_ohe = train_test_split(df_ohe, test_size=test_split, random_state=random_state)
    train_df, test_df = train_test_split(df, test_size=test_split, random_state=random_state)
    train_df.to_csv(cfg['PATHS']['TRAIN_SET'], sep=',', header=True, index=False) # Save train & test set for LIME
    test_df.to_csv(cfg['PATHS']['TEST_SET'], sep=',', header=True, index=False)
    relative_val_split = val_split / (train_split + val_split)  # Calculate fraction of train_df to be used for validation
    train_df_ohe, val_df_ohe = train_test_split(train_df_ohe, test_size=relative_val_split)

    # Anonymize clients
    train_df_ohe.drop('ClientID', axis=1, inplace=True)
    val_df_ohe.drop('ClientID', axis=1, inplace=True)
    test_df_ohe.drop('ClientID', axis=1, inplace=True)

    # Get indices of noncategorical features
    noncat_feat_idxs = [test_df_ohe.columns.get_loc(c) for c in noncat_features if c in test_df_ohe]

    # Separate ground truth from dataframe and convert to numpy arrays
    data['Y_train'] = np.array(train_df_ohe.pop('GroundTruth'))
    data['Y_val'] = np.array(val_df_ohe.pop('GroundTruth'))
    data['Y_test'] = np.array(test_df_ohe.pop('GroundTruth'))

    # Convert feature dataframes to numpy arrays
    data['X_train'] = np.array(train_df_ohe)
    data['X_val'] = np.array(val_df_ohe)
    data['X_test'] = np.array(test_df_ohe)

    # Normalize numerical data and save the scaler for prediction.
    col_trans_scaler = ColumnTransformer(transformers=[('col_trans_ordinal', StandardScaler(), noncat_feat_idxs)],
                                         remainder='passthrough')
    data['X_train'] = col_trans_scaler.fit_transform(data['X_train'])   # Only fit train data to prevent data leakage
    data['X_val'] = col_trans_scaler.transform(data['X_val'])
    data['X_test'] = col_trans_scaler.transform(data['X_test'])
    dump(col_trans_scaler, cfg['PATHS']['SCALER_COL_TRANSFORMER'], compress=True)
    return data


def load_time_series_dataset(cfg, slide=None):
    '''
    Load the static and time series data from disk and join them. Create time series examples to form large dataset with
    time series and static features. Partition into train/val/test sets. Normalize numerical data.
    :param cfg: Project config (from config.yml)
    :param slide: Int that controls how many recent dates to cut off from the dataset
    :return: A dict of partitioned and normalized datasets, split into examples and labels
    '''

    # Load data info generated during preprocessing
    data = {}
    data['METADATA'] = {}
    input_stream = open(cfg['PATHS']['DATA_INFO'], 'r')
    data_info = yaml.full_load(input_stream)
    data['METADATA']['N_WEEKS'] = data_info['N_WEEKS']
    noncat_features = data_info['NON_CAT_FEATURES']   # Noncategorical features to be scaled
    T_X = cfg['DATA']['TIME_SERIES']['T_X']
    tqdm.pandas()

    # Load data (before and after one-hot encoding)
    df_ohe = pd.read_csv(cfg['PATHS']['PROCESSED_OHE_DATA'])
    df = pd.read_csv(cfg['PATHS']['PROCESSED_DATA'])   # Static and dynamic data prior to one hot encoding
    time_series_feats = [f for f in df.columns if ('-Day_' in f) and (')' not in f)]

    # Partition dataset by date
    unique_dates = np.flip(df_ohe['Date'].unique()).flatten()
    val_split = cfg['TRAIN']['VAL_SPLIT']
    if val_split*unique_dates.shape[0] < 1:
        val_split = 1.0 / unique_dates.shape[0]     # Ensure validation set contains records from at least 1 time step
        print("Val set split in config.yml is too small. Increased to " + str(val_split))
    test_split = cfg['TRAIN']['TEST_SPLIT']
    if test_split*unique_dates.shape[0] < 1:
        test_split = 1.0 / unique_dates.shape[0]    # Ensure test set contains records from at least 1 time step
        print("Test set split in config.yml is too small. Increased to " + str(test_split))
    if slide is None:
        test_df_dates = unique_dates[-int(test_split*unique_dates.shape[0]):]
        val_df_dates = unique_dates[-int((test_split + val_split)*unique_dates.shape[0]):-int(test_split*unique_dates.shape[0])]
        train_df_dates = unique_dates[0:-int((test_split + val_split)*unique_dates.shape[0])]
    else:
        test_split_size = max(int((test_split) * unique_dates.shape[0]), 1)
        val_split_size = max(int((val_split) * unique_dates.shape[0]), 1)
        offset = slide * test_split_size
        if offset == 0:
            test_df_dates = unique_dates[-(test_split_size):]
        else:
            test_df_dates = unique_dates[-(test_split_size + offset):-offset]
        val_df_dates = unique_dates[-(val_split_size + test_split_size + offset):-(test_split_size + offset)]
        train_df_dates = unique_dates[0:-(val_split_size + test_split_size + offset)]

    train_df_ohe = df_ohe[df_ohe['Date'].isin(train_df_dates)]
    val_df_ohe = df_ohe[df_ohe['Date'].isin(val_df_dates)]
    test_df_ohe = df_ohe[df_ohe['Date'].isin(test_df_dates)]
    train_df = df[df['Date'].isin(train_df_dates)]
    val_df = df[df['Date'].isin(val_df_dates)]
    test_df = df[df['Date'].isin(test_df_dates)]
    print('Train set size = ' + str(train_df_ohe.shape[0]) + '. Val set size = ' + str(val_df_ohe.shape[0]) +
          '. Test set size = ' + str(test_df_ohe.shape[0]))

    # Save train & test set for LIME
    train_df.to_csv(cfg['PATHS']['TRAIN_SET'], sep=',', header=True, index=False)
    val_df.to_csv(cfg['PATHS']['VAL_SET'], sep=',', header=True, index=False)
    test_df.to_csv(cfg['PATHS']['TEST_SET'], sep=',', header=True, index=False)

    # Anonymize clients
    train_df_ohe.drop(['ClientID', 'Date'], axis=1, inplace=True)
    val_df_ohe.drop(['ClientID', 'Date'], axis=1, inplace=True)
    test_df_ohe.drop(['ClientID', 'Date'], axis=1, inplace=True)

    # Get indices of noncategorical features
    noncat_feat_idxs = [test_df_ohe.columns.get_loc(c) for c in noncat_features if c in test_df_ohe]

    # Separate ground truth from dataframe and convert to numpy arrays
    data['Y_train'] = np.array(train_df_ohe.pop('GroundTruth'))
    data['Y_val'] = np.array(val_df_ohe.pop('GroundTruth'))
    data['Y_test'] = np.array(test_df_ohe.pop('GroundTruth'))

    # Convert feature dataframes to numpy arrays
    data['X_train'] = np.array(train_df_ohe)
    data['X_val'] = np.array(val_df_ohe)
    data['X_test'] = np.array(test_df_ohe)

    # Normalize numerical data and save the scaler for prediction.
    col_trans_scaler = ColumnTransformer(transformers=[('col_trans_ordinal', StandardScaler(), noncat_feat_idxs)],
                                         remainder='passthrough')
    data['X_train'] = col_trans_scaler.fit_transform(data['X_train'])   # Only fit train data to prevent data leakage
    data['X_val'] = col_trans_scaler.transform(data['X_val'])
    data['X_test'] = col_trans_scaler.transform(data['X_test'])
    dump(col_trans_scaler, cfg['PATHS']['SCALER_COL_TRANSFORMER'], compress=True)

    data['METADATA']['NUM_TS_FEATS'] = len(time_series_feats)   # Number of different time series features
    data['METADATA']['T_X'] = T_X
    return data


def define_callbacks(cfg):
    '''
    Build a list of Keras callbacks for training a model.
    :param cfg: Project config object
    :return: a list of Keras callbacks
    '''
    early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=cfg['TRAIN']['PATIENCE'], mode='min', restore_best_weights=True)
    callbacks = [early_stopping]
    return callbacks


def train_model(cfg, data, callbacks, verbose=2):
    '''
    Train a and evaluate model on given data.
    :param cfg: Project config (from config.yml)
    :param data: dict of partitioned dataset
    :param callbacks: list of callbacks for Keras model
    :param verbose: Verbosity mode to pass to model.fit()
    :return: Trained model and associated performance metrics on the test set
    '''

    # Apply class imbalance strategy
    num_neg, num_pos = np.bincount(data['Y_train'].astype(int))
    class_weight = None
    if cfg['TRAIN']['IMB_STRATEGY'] == 'class_weight':
        class_weight = get_class_weights(num_pos, num_neg, cfg['TRAIN']['POS_WEIGHT'])
    elif cfg['TRAIN']['IMB_STRATEGY'] != 'none':
        data['X_train'], data['Y_train'] = minority_oversample(data['X_train'], data['Y_train'],
                                                               algorithm=cfg['TRAIN']['IMB_STRATEGY'])

    thresholds = cfg['TRAIN']['THRESHOLDS']     # Load classification thresholds

    # List metrics
    metrics = [BinaryAccuracy(name='accuracy'), Precision(name='precision', thresholds=thresholds),
               Recall(name='recall', thresholds=thresholds), F1Score(name='f1score', thresholds=thresholds),
               AUC(name='auc')]

    # Compute output bias
    num_neg, num_pos = np.bincount(data['Y_train'].astype(int))
    output_bias = np.log([num_pos / num_neg])

    # Build the model graph.
    if cfg['TRAIN']['MODEL_DEF'] == 'hifis_rnn_mlp':
        model_def = hifis_rnn_mlp
    elif cfg['TRAIN']['MODEL_DEF'] == 'hifis_mlp':
        model_def = hifis_mlp
    elif cfg['TRAIN']['MODEL_DEF'] == 'logistic_regression':
        model_def = logistic_regression
    elif cfg['TRAIN']['MODEL_DEF'] == 'random_forest':
        model_def = random_forest
    else:
        model_def = xgboost_model
    model = model_def(cfg['MODELS'][cfg['TRAIN']['MODEL_DEF'].upper()], input_dim=(data['X_train'].shape[-1],), metrics=metrics,
                      metadata=data['METADATA'], output_bias=output_bias)

    # Train the model.
    history = model.fit(data['X_train'], data['Y_train'], batch_size=cfg['TRAIN']['BATCH_SIZE'],
                        epochs=cfg['TRAIN']['EPOCHS'], validation_data=(data['X_val'], data['Y_val']),
                        callbacks=callbacks, class_weight=class_weight, verbose=verbose)

    # Run the model on the test set and print the resulting performance metrics.
    test_results = model.evaluate(data['X_test'], data['Y_test'])
    test_metrics = {}
    test_summary_str = [['**Metric**', '**Value**']]
    for metric, value in zip(model.metrics_names, test_results):
        test_metrics[metric] = value
        print(metric, ' = ', value)
        test_summary_str.append([metric, str(value)])
    return model, test_metrics


def multi_train(cfg, data, callbacks, base_log_dir):
    '''
    Trains a model a series of times and returns the model with the best test set metric (specified in cfg)
    :param cfg: Project config (from config.yml)
    :param data: Partitioned dataset
    :param callbacks: List of callbacks to pass to model.fit()
    :param base_log_dir: Base directory to write logs
    :return: The trained Keras model with best test set performance on the metric specified in cfg
    '''

    # Load order of metric preference
    metric_preference = cfg['TRAIN']['METRIC_PREFERENCE']
    best_metrics = dict.fromkeys(metric_preference, 0.0)
    if 'loss' in metric_preference:
        best_metrics['loss'] = 10000.0

    # Create dict to store test set metrics for all models
    test_metrics_dict = dict.fromkeys(['Model #'] + metric_preference, [])

    # Train NUM_RUNS models and return the best one according to the preferred metrics
    for i in range(cfg['TRAIN']['NUM_RUNS']):
        print("Training run ", i+1, " / ", cfg['TRAIN']['NUM_RUNS'])
        cur_callbacks = callbacks.copy()
        cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        if base_log_dir is not None:
            log_dir = base_log_dir + cur_date
            cur_callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))

        # Train the model and evaluate performance on test set
        new_model, test_metrics = train_model(cfg, data, cur_callbacks)

        # Record this model's test set metrics
        test_metrics_dict['Model #'] = test_metrics_dict['Model #'] + [i + 1]
        for metric in metric_preference:
            test_metrics_dict[metric] = test_metrics_dict[metric] + [test_metrics[metric]]

        # Log test set results and images
        if base_log_dir is not None:
            log_test_results(cfg, new_model, data, test_metrics, log_dir)

        # If this model outperforms the previous ones based on the specified metric preferences, save this one.
        for i in range(len(metric_preference)):
            if (((metric_preference[i] == 'loss') and (test_metrics[metric_preference[i]] < best_metrics[metric_preference[i]]))
                    or ((metric_preference[i] != 'loss') and (test_metrics[metric_preference[i]] > best_metrics[metric_preference[i]]))):
                best_model = new_model
                best_metrics = test_metrics
                best_model_date = cur_date
                break
            elif (test_metrics[metric_preference[i]] == best_metrics[metric_preference[i]]):
                continue
            else:
                break

    print("Best model test metrics: ", best_metrics)
    return best_model, best_metrics, test_metrics_dict, best_model_date

def random_hparam_search(cfg, data, callbacks, log_dir):
    '''
    Conduct a random hyperparameter search over the ranges given for the hyperparameters in config.yml and log results
    in TensorBoard. Model is trained x times for y random combinations of hyperparameters.
    :param cfg: Project config
    :param data: Dict containing the partitioned datasets
    :param callbacks: List of callbacks for Keras model (excluding TensorBoard)
    :param log_dir: Base directory in which to store logs
    '''

    # Define HParam objects for each hyperparameter we wish to tune.
    hp_ranges = cfg['TRAIN']['HP']['RANGES']
    HPARAMS = []
    HPARAMS.append(hp.HParam('NODES0', hp.Discrete(hp_ranges['NODES0'])))
    HPARAMS.append(hp.HParam('NODES1', hp.Discrete(hp_ranges['NODES1'])))
    HPARAMS.append(hp.HParam('LAYERS', hp.Discrete(hp_ranges['LAYERS'])))
    HPARAMS.append(hp.HParam('DROPOUT', hp.RealInterval(hp_ranges['DROPOUT'][0], hp_ranges['DROPOUT'][1])))
    HPARAMS.append(hp.HParam('L2_LAMBDA', hp.RealInterval(hp_ranges['L2_LAMBDA'][0], hp_ranges['L2_LAMBDA'][1])))
    HPARAMS.append(hp.HParam('LR', hp.RealInterval(hp_ranges['LR'][0], hp_ranges['LR'][1])))
    HPARAMS.append(hp.HParam('BETA_1', hp.RealInterval(hp_ranges['BETA_1'][0], hp_ranges['BETA_1'][1])))
    HPARAMS.append(hp.HParam('BETA_2', hp.RealInterval(hp_ranges['BETA_2'][0], hp_ranges['BETA_2'][1])))
    HPARAMS.append(hp.HParam('OPTIMIZER', hp.Discrete(hp_ranges['OPTIMIZER'])))
    HPARAMS.append(hp.HParam('BATCH_SIZE', hp.Discrete(hp_ranges['BATCH_SIZE'])))
    HPARAMS.append(hp.HParam('POS_WEIGHT', hp.RealInterval(hp_ranges['POS_WEIGHT'][0], hp_ranges['POS_WEIGHT'][1])))
    HPARAMS.append(hp.HParam('IMB_STRATEGY', hp.Discrete(hp_ranges['IMB_STRATEGY'])))
    if cfg['TRAIN']['MODEL_DEF'] == 'hifis_rnn_mlp':
        HPARAMS.append(hp.HParam('LSTM_UNITS', hp.Discrete(hp_ranges['LSTM_UNITS'])))

    # Define test set metrics that we wish to log to TensorBoard for each training run
    HP_METRICS = [hp.Metric(metric, display_name='Test ' + metric) for metric in cfg['TRAIN']['HP']['METRICS']]

    # Configure TensorBoard to log the results
    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams_config(hparams=HPARAMS, metrics=HP_METRICS)

    # Complete a number of training runs at different hparam values and log the results.
    repeats_per_combo = cfg['TRAIN']['HP']['REPEATS']   # Number of times to train the model per combination of hparams
    num_combos = cfg['TRAIN']['HP']['COMBINATIONS']     # Number of random combinations of hparams to attempt
    num_sessions = num_combos * repeats_per_combo       # Total number of runs in this experiment
    trial_id = 0
    for group_idx in range(num_combos):
        rand = random.Random()
        HPARAMS = {h: h.domain.sample_uniform(rand) for h in HPARAMS}
        hparams = {h.name: HPARAMS[h] for h in HPARAMS}  # To pass to model definition
        for repeat_idx in range(repeats_per_combo):
            trial_id += 1
            print("Running training session %d/%d" % (trial_id, num_sessions))
            print("Hparam values: ", {h.name: HPARAMS[h] for h in HPARAMS})
            trial_logdir = os.path.join(log_dir, str(trial_id))     # Need specific logdir for each trial
            callbacks_hp = callbacks + [TensorBoard(log_dir=trial_logdir, profile_batch=0, write_graph=False)]

            # Set values of hyperparameters for this run in config file.
            for h in hparams:
                if h in ['LR', 'L2_LAMBDA']:
                    val = 10 ** hparams[h]      # These hyperparameters are sampled on the log scale.
                else:
                    val = hparams[h]
                cfg['MODELS'][cfg['TRAIN']['MODEL_DEF'].upper()][h] = val

            # Set some hyperparameters that are not specified in model definition.
            cfg['TRAIN']['BATCH_SIZE'] = hparams['BATCH_SIZE']
            cfg['TRAIN']['IMB_STRATEGY'] = hparams['IMB_STRATEGY']
            cfg['TRAIN']['POS_WEIGHT'] = hparams['POS_WEIGHT']

            # Run a training session and log the performance metrics on the test set to HParams dashboard in TensorBoard
            with tf.summary.create_file_writer(trial_logdir).as_default():
                hp.hparams(HPARAMS, trial_id=str(trial_id))
                model, test_metrics = train_model(cfg, data, callbacks_hp, verbose=2)    # Train model
                for metric in HP_METRICS:
                    if metric._tag in test_metrics:
                        tf.summary.scalar(metric._tag, test_metrics[metric._tag], step=1)   # Log test metric
    return


def kfold_cross_validation(cfg, callbacks, base_log_dir):
    '''
    Perform k-fold cross-validation for the HIFIS-MLP model. Data is saved in CSV format to the
    "/results/experiments" folder.
    :param cfg: Project config dict
    :param callbacks: List of Keras callbacks
    :param base_log_dir: base log directory for TensorBoard logs
    '''
    df_ohe = pd.read_csv(cfg['PATHS']['PROCESSED_OHE_DATA'])
    data_info = yaml.full_load(open(cfg['PATHS']['DATA_INFO'], 'r'))
    metadata = {'N_WEEKS': data_info['N_WEEKS']}
    data = {}
    data['METADATA'] = metadata
    noncat_features = data_info['NON_CAT_FEATURES']  # Noncategorical features to be scaled
    thresholds = cfg['TRAIN']['THRESHOLDS']

    k = cfg['DATA']['KFOLDS']     # i.e. "k" for nested cross validation
    val_split = 1.0 / k
    train_split = 1.0 - 2.0 / k     # Let val set be same size as test set

    metrics_list = cfg['TRAIN']['METRIC_PREFERENCE']
    metrics_df = pd.DataFrame(np.zeros((k + 2, len(metrics_list) + 1)), columns=['Fold'] + metrics_list)
    metrics_df['Fold'] = list(range(1, k + 1)) + ['mean', 'std']

    df_ohe.drop('ClientID', axis=1, inplace=True)     # Anonymize clients
    noncat_feat_idxs = [df_ohe.columns.get_loc(c) for c in noncat_features if c in df_ohe]
    Y = np.array(df_ohe.pop('GroundTruth'))
    X = np.array(df_ohe)

    k_fold = KFold(n_splits=k, shuffle=True)
    row_idx = 0
    for train_indices, test_indices in k_fold.split(df_ohe):
        X_train = X[train_indices]
        Y_train = Y[train_indices]
        X_test = X[test_indices]
        Y_test = Y[test_indices]

        random_state = np.random.randint(0, high=1000)
        relative_val_split = val_split / (train_split + val_split)  # Calculate fraction of train set to be used for validation
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=relative_val_split,
                                                         random_state=random_state)

        # Normalize numerical data and save the scaler for prediction.
        col_trans_scaler = ColumnTransformer(transformers=[('col_trans_ordinal', StandardScaler(), noncat_feat_idxs)],
                                             remainder='passthrough')
        data['X_train'] = col_trans_scaler.fit_transform(X_train)  # Only fit train data to prevent data leakage
        data['X_val'] = col_trans_scaler.transform(X_val)
        data['X_test'] = col_trans_scaler.transform(X_test)
        data['Y_train'] = Y_train
        data['Y_val'] = Y_val
        data['Y_test'] = Y_test

        cur_callbacks = callbacks.copy()
        cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        if base_log_dir is not None:
            log_dir = base_log_dir + cur_date
            cur_callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))

        # Train the model and evaluate performance on test set
        new_model, test_metrics = train_model(cfg, data, cur_callbacks)

        for metric in test_metrics:
            if any(metric in c for c in metrics_df.columns):
                if any(metric in m for m in ['precision', 'recall', 'f1score']) and isinstance(thresholds, list):
                    for j in range(len(thresholds)):
                        metrics_df[metric + '_thr=' + str(thresholds[j])][row_idx] = test_metrics[metric][j]
                else:
                    metrics_df[metric][row_idx] = test_metrics[metric]
        row_idx += 1

        # Log test set results and images
        if base_log_dir is not None:
            log_test_results(cfg, new_model, data, test_metrics, log_dir)

    # Record mean and standard deviation of test set results
    for metric in metrics_list:
        metrics_df[metric][k] = metrics_df[metric][0:-2].mean()
        metrics_df[metric][k + 1] = metrics_df[metric][0:-2].std()

    # Save results
    experiment_path = cfg['PATHS']['EXPERIMENTS'] + 'kFoldCV' + cur_date + '.csv'
    metrics_df.to_csv(experiment_path, columns=metrics_df.columns, index_label=False, index=False)
    return metrics_df


def nested_cross_validation(cfg, callbacks, base_log_dir):
    '''
    Perform nested cross-validation with day-forward chaining for the HIFIS-RNN-MLP model. Data is saved in CSV format
    to the results/experiments/ folder.
    :param cfg: Project config dict
    :param callbacks: List of Keras callbacks
    :param base_log_dir: base log directory for TensorBoard logs
    '''

    num_folds = cfg['DATA']['TIME_SERIES']['FOLDS']     # i.e. "k" for nested cross validation
    metrics = cfg['TRAIN']['METRIC_PREFERENCE']
    thresholds = cfg['TRAIN']['THRESHOLDS']
    metrics_list = []
    for m in metrics:
        if m in ['precision', 'recall', 'f1score'] and isinstance(thresholds, list):
            metrics_list += [(m + '_thr=' + str(t)) for t in thresholds]
        else:
            metrics_list.append(m)
    metrics_df = pd.DataFrame(np.zeros((num_folds + 2, len(metrics_list) + 1)), columns=['Fold'] + metrics_list)
    metrics_df['Fold'] = list(range(1, num_folds + 1)) + ['mean', 'std']

    # Train a model k times with different folds
    for i in range(num_folds):
        data = load_time_series_dataset(cfg, slide=i)
        cur_callbacks = callbacks.copy()
        cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        if base_log_dir is not None:
            log_dir = base_log_dir + cur_date
            cur_callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))

        # Train the model and evaluate performance on test set
        new_model, test_metrics = train_model(cfg, data, cur_callbacks)
        for metric in test_metrics:
            if any(metric in c for c in metrics_df.columns):
                if any(metric in m for m in ['precision', 'recall', 'f1score']) and isinstance(thresholds, list):
                    for j in range(len(thresholds)):
                        metrics_df[metric + '_thr=' + str(thresholds[j])][i] = test_metrics[metric][j]
                else:
                    metrics_df[metric][i] = test_metrics[metric]

        # Log test set results and images
        if base_log_dir is not None:
            log_test_results(cfg, new_model, data, test_metrics, log_dir)

    # Record mean and standard deviation of test set results
    for metric in metrics_list:
        metrics_df[metric][num_folds] = metrics_df[metric][0:-2].mean()
        metrics_df[metric][num_folds + 1] = metrics_df[metric][0:-2].std()

    # Save results
    experiment_path = cfg['PATHS']['EXPERIMENTS'] + 'nestedCV' + cur_date + '.csv'
    metrics_df.to_csv(experiment_path, columns=metrics_df.columns, index_label=False, index=False)
    return metrics_df


def log_test_results(cfg, model, data, test_metrics, log_dir):
    '''
    Visualize performance of a trained model on the test set. Optionally save the model.
    :param cfg: Project config
    :param model: A trained Keras model
    :param data: Dict containing datasets
    :param test_metrics: Dict of test set performance metrics
    :param log_dir: Path to write TensorBoard logs
    '''

    # Visualization of test results
    test_predictions = model.predict(data['X_test'], batch_size=cfg['TRAIN']['BATCH_SIZE'])
    plt = plot_roc("Test set", data['Y_test'], test_predictions, dir_path=None)
    roc_img = plot_to_tensor()
    plt = plot_confusion_matrix(data['Y_test'], test_predictions, dir_path=None)
    cm_img = plot_to_tensor()

    # Log test set results and plots in TensorBoard
    writer = tf.summary.create_file_writer(logdir=log_dir)

    # Create table of test set metrics
    test_summary_str = [['**Metric**','**Value**']]
    thresholds = cfg['TRAIN']['THRESHOLDS']  # Load classification thresholds
    for metric in test_metrics:
        if metric in ['precision', 'recall'] and isinstance(metric, list):
            metric_values = dict(zip(thresholds, test_metrics[metric]))
        else:
            metric_values = test_metrics[metric]
        test_summary_str.append([metric, str(metric_values)])

    # Create table of model and train config values
    hparam_summary_str = [['**Variable**', '**Value**']]
    for key in cfg['TRAIN']:
        hparam_summary_str.append([key, str(cfg['TRAIN'][key])])
    if cfg['MODELS'][cfg['TRAIN']['MODEL_DEF'].upper()] is not None:
        for key in cfg['MODELS'][cfg['TRAIN']['MODEL_DEF'].upper()]:
            hparam_summary_str.append([key, str(cfg['MODELS'][cfg['TRAIN']['MODEL_DEF'].upper()][key])])

    # Write to TensorBoard logs
    with writer.as_default():
        tf.summary.text(name='Metrics - Test Set', data=tf.convert_to_tensor(test_summary_str), step=0)
        tf.summary.text(name='Model Hyperparameters', data=tf.convert_to_tensor(hparam_summary_str), step=0)
        tf.summary.image(name='ROC Curve (Test Set)', data=roc_img, step=0)
        tf.summary.image(name='Confusion Matrix (Test Set)', data=cm_img, step=0)
    return


def train_experiment(cfg=None, experiment='single_train', save_weights=True, write_logs=True):
    '''
    Defines and trains HIFIS-v2 model. Prints and logs relevant metrics.
    :param cfg: Project config dictionary
    :param experiment: The type of training experiment. Choices are {'single_train', 'multi_train', 'hparam_search'}
    :param save_weights: A flag indicating whether to save the model weights
    :param write_logs: A flag indicating whether to write TensorBoard logs
    :return: A dictionary of metrics on the test set
    '''

    # Load project config data
    if cfg is None:
        cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

    # Set logs directory
    cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = cfg['PATHS']['LOGS'] + "training/" + cur_date if write_logs else None
    if not os.path.exists(cfg['PATHS']['LOGS'] + "training/"):
        os.makedirs(cfg['PATHS']['LOGS'] + "training/")

    # Load preprocessed data and partition into training, validation and test sets.
    if cfg['TRAIN']['DATASET_TYPE'] == 'static_and_dynamic':
        data = load_time_series_dataset(cfg)
    else:
        data = load_dataset(cfg)

    # Set callbacks
    callbacks = define_callbacks(cfg)

    # Conduct the desired train experiment
    if experiment == 'hparam_search':
        log_dir = cfg['PATHS']['LOGS'] + "hparam_search/" + cur_date
        random_hparam_search(cfg, data, callbacks, log_dir)
    elif experiment == 'cross_validation':
        base_log_dir = cfg['PATHS']['LOGS'] + "training/" if write_logs else None
        if cfg['TRAIN']['DATASET_TYPE'] == 'static_and_dynamic':
            _ = nested_cross_validation(cfg, callbacks, base_log_dir)   # If time series data, do nested CV
        else:
            _ = kfold_cross_validation(cfg, callbacks, base_log_dir)    # If not time series data, do k-fold CV
    else:
        if experiment == 'multi_train':
            base_log_dir = cfg['PATHS']['LOGS'] + "training/" if write_logs else None
            model, test_metrics, test_metrics_dict, cur_date = multi_train(cfg, data, callbacks, base_log_dir)
            test_set_metrics_df = pd.DataFrame(test_metrics_dict)
            test_set_metrics_df.to_csv(cfg['PATHS']['MULTI_TRAIN_TEST_METRICS'].split('.')[0] + cur_date + '.csv')
        else:
            if write_logs:
                tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
                callbacks.append(tensorboard)
            model, test_metrics = train_model(cfg, data, callbacks)
            if write_logs:
                log_test_results(cfg, model, data, test_metrics, log_dir)
        if save_weights:
            model_path = cfg['PATHS']['MODEL_WEIGHTS'] + 'model' + cur_date + '.h5'
            save_model(model, model_path)  # Save the model's weights

    return

if __name__ == '__main__':
    cfg = yaml.full_load(open("./config.yml", 'r'))
    train_experiment(cfg=cfg, experiment=cfg['TRAIN']['EXPERIMENT'], save_weights=True, write_logs=True)

