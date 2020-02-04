import pandas as pd
import yaml
import os
import random
import tensorflow as tf
from six.moves import xrange
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.externals.joblib import dump
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorboard.plugins.hparams import api as hp
from src.models.models import model1
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

def minority_oversample(X_train, Y_train):
    '''
    Oversample the minority class by duplicating some of its samples
    :param X_train: Training set features
    :param Y_train: Training set labels
    :return: A new training set containing oversampled examples
    '''
    ros = RandomOverSampler(random_state=np.random.randint(0, high=1000))
    X_resampled, Y_resampled = ros.fit_resample(X_train, Y_train)
    print("Train set shape before oversampling: ", X_train.shape, " Train set shape after resampling: ", X_resampled.shape)
    return X_resampled, Y_resampled

def smote(X_train, Y_train):
    smote = SMOTE(random_state=np.random.randint(0, high=1000))
    X_resampled, Y_resampled = smote.fit_resample(X_train, Y_train)
    print("Train set shape before SMOTE: ", X_train.shape, " Train set shape after SMOTE: ", X_resampled.shape)
    return X_resampled, Y_resampled

def adasyn(X_train, Y_train):
    adasyn = ADASYN(random_state=np.random.randint(0, high=1000))
    X_resampled, Y_resampled = adasyn.fit_resample(X_train, Y_train)
    print("Train set shape before ADASYN: ", X_train.shape, " Train set shape after ADASYN: ", X_resampled.shape)
    return X_resampled, Y_resampled

def load_dataset(cfg):

    # Load config data generated from preprocessing
    input_stream = open(os.getcwd() + cfg['PATHS']['INTERPRETABILITY'], 'r')
    feature_info = yaml.full_load(input_stream)
    noncat_features = feature_info['NON_CAT_FEATURES']   # Noncategorical features to be scaled

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
    data = {}
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

def train_model(cfg, data, model, callbacks, verbose=1):

    # Oversample minority class
    if cfg['TRAIN']['IMB_STRATEGY'] == 'minority_oversample':
        data['X_train'], data['Y_train'] = minority_oversample(data['X_train'], data['Y_train'])
    elif cfg['TRAIN']['IMB_STRATEGY'] == 'smote':
        data['X_train'], data['Y_train'] = smote(data['X_train'], data['Y_train'])
    elif cfg['TRAIN']['IMB_STRATEGY'] == 'adasyn':
        data['X_train'], data['Y_train'] = adasyn(data['X_train'], data['Y_train'])

    # Calculate class weights
    num_neg, num_pos = np.bincount(data['Y_train'].astype(int))
    class_weight = None
    if cfg['TRAIN']['IMB_STRATEGY'] == 'class_weight':
        class_weight = get_class_weights(num_pos, num_neg, cfg['TRAIN']['POS_WEIGHT'])

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


def multi_train(cfg, data, model, callbacks):
    # Train model for a specified number of times and keep the one with the best target test metric
    metric_monitor = cfg['TRAIN']['METRIC_MONITOR']
    best_value = 1000.0 if metric_monitor == 'loss' else 0.0
    for i in range(cfg['TRAIN']['NUM_RUNS']):
        print("Training run ", i+1, " / ", cfg['TRAIN']['NUM_RUNS'])

        # Train the model and evaluate performance on test set
        model, test_metrics = train_model(cfg, data, model, callbacks)

        # If this model outperforms the previous ones based on the specified metric, save this one.
        if (((metric_monitor == 'loss') and (test_metrics[metric_monitor] < best_value))
                or ((metric_monitor != 'loss') and (test_metrics[metric_monitor] > best_value))):
            best_value = test_metrics[metric_monitor]
            best_model = model
            best_metrics = test_metrics
    print("Best model test metrics: ", best_metrics)
    return best_model, best_metrics

def random_hparam_search(cfg, data, metrics, shape, callbacks, log_dir):
    '''
    Conduct a random hyperparameter search over the ranges given for the hyperparameters in config.yml and log results
    in TensorBoard. Model is trained x times for y random combinations of hyperparameters.
    :param cfg: Project config
    :param data: Dict containing the partitioned datasets
    :param metrics: List of model metrics
    :param shape: Shape of input examples
    :param callbacks: List of callbacks (excluding TensorBoard)
    :param log_dir: Base directory in which to store logs
    :return: (Last model trained, esultant test set metrics)
    '''

    # Define HParam objects for each hyperparameter we wish to tune.
    hp_ranges = cfg['TRAIN']['HP']['RANGES']
    HPARAMS = []
    HPARAMS.append(hp.HParam('NODES', hp.Discrete(hp_ranges['NODES'])))
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

    # Define metrics that we wish to log to TensorBoard for each training run
    HP_METRICS = [hp.Metric('epoch_' + metric, group='validation', display_name='Val ' + metric) for metric in cfg['TRAIN']['HP']['METRICS']]
    HP_METRICS.append(hp.Metric('epoch_loss', group='train', display_name='Train loss'))    # Help catch overfitting

    # Configure TensorBoard to log the results
    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams_config(hparams=HPARAMS, metrics=HP_METRICS)

    # Complete a number of training runs at different hparam values and log the results.
    repeats_per_combo = cfg['TRAIN']['HP']['REPEATS']   # Number of times to train the model per combination of hparams
    num_combos = cfg['TRAIN']['HP']['COMBINATIONS']     # Number of random combinations of hparams to attempt
    num_sessions = num_combos * repeats_per_combo       # Total number of runs in this experiment
    trial_id = 0
    for group_idx in xrange(num_combos):
        rand = random.Random()
        hparams = {h.name: h.domain.sample_uniform(rand) for h in HPARAMS}  # To pass to model definition
        HPARAMS = {h: h.domain.sample_uniform(rand) for h in HPARAMS}
        for repeat_idx in xrange(repeats_per_combo):
            trial_id += 1
            print("Running training session %d/%d" % (trial_id, num_sessions))
            print("Hparam values: ", {h.name: HPARAMS[h] for h in HPARAMS})
            trial_logdir = os.path.join(log_dir, str(trial_id))     # Need specific logdir for each trial
            callbacks_hp = callbacks + [TensorBoard(log_dir=trial_logdir, profile_batch=0, write_graph=False),
                                        hp.KerasCallback(trial_logdir, hparams, trial_id=str(trial_id))]
            model = model1(cfg['NN']['MODEL1'], shape, metrics, hparams)
            cfg['TRAIN']['BATCH_SIZE'] = hparams['BATCH_SIZE']  # This hparam is not set in model definition
            cfg['TRAIN']['POS_WEIGHT'] = hparams['POS_WEIGHT']  # This hparam is not set in model definition
            if hparams['IMB_STRATEGY'] == 'class_weight':
                cfg['CLASS_WEIGHT'] = True
            if hparams['IMB_STRATEGY'] == 'min_oversample':
                cfg['MINORITY_OVERSAMPLE'] = True
            if hparams['IMB_STRATEGY'] == 'smote':
                cfg['SMOTE'] = True
            model, test_metrics = train_model(cfg, data, model, callbacks_hp, verbose=2)
    return model, test_metrics

def train_experiment(save_weights=True, write_logs=True):
    '''
    Defines and trains HIFIS-v2 model. Prints and logs relevant metrics.
    :param save_weights: A flag indicating whether to save the model weights
    :return: A dictionary of metrics on the test set
    '''

    # Load project config data
    input_stream = open(os.getcwd() + "/config.yml", 'r')
    cfg = yaml.full_load(input_stream)

    # Load preprocessed data and partition into training, validation and test sets.
    data = load_dataset(cfg)

    plot_path = cfg['PATHS']['IMAGES']  # Path for images of matplotlib figures
    cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    # Define metrics.
    metrics = ['accuracy', BinaryAccuracy(name='accuracy'), Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]

    # Set callbacks.
    early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=15, mode='min', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)
    callbacks = [early_stopping]
    if write_logs:
        log_dir = cfg['PATHS']['LOGS'] + cur_date
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard)

    # Define the model.
    model = model1(cfg['NN']['MODEL1'], (data['X_train'].shape[-1],), metrics)   # Build model graph

    # Train a model
    model, test_metrics = train_model(cfg, data, model, callbacks)
    #model, test_metrics = multi_train(cfg, data, model, callbacks)
    #model, test_metrics = random_hparam_search(cfg, data, metrics, (data['X_train'].shape[-1],), [callbacks[0]], log_dir)

    # Visualization of test results
    test_predictions = model.predict(data['X_test'], batch_size=cfg['TRAIN']['BATCH_SIZE'])
    roc_img = plot_roc("Test set", data['Y_test'], test_predictions, file_path=None)
    cm_img = plot_confusion_matrix(data['Y_test'], test_predictions, file_path=None)

    # Log test set results and plots in TensorBoard
    if write_logs:
        writer = tf.summary.create_file_writer(logdir=log_dir)
        test_summary_str = [['**Metric**','**Value**']]
        for metric in test_metrics:
            test_summary_str.append([metric, str(test_metrics[metric])])
        with writer.as_default():
            tf.summary.text(name='Test set metrics', data=tf.convert_to_tensor(test_summary_str), step=0)
            tf.summary.image(name='ROC Curve (Test Set)', data=roc_img, step=0)
            tf.summary.image(name='Confusion Matrix (Test Set)', data=cm_img, step=0)

    if save_weights:
        model_path = os.path.splitext(cfg['PATHS']['MODEL_WEIGHTS'])[0] + cur_date + '.h5'
        save_model(model, model_path)        # Save model weights
    return test_metrics

if __name__ == '__main__':
    results = train_experiment(save_weights=True, write_logs=True)

