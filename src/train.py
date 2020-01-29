import pandas as pd
import yaml
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.externals.joblib import dump
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from src.models.models import model1
from src.custom.metrics import F1Score
from src.visualization.visualize import *

def get_class_weights(num_pos, num_neg):
    '''
    Computes weights for each class to be applied in the loss function during training.
    :param num_pos: # positive samples
    :param num_neg: # negative samples
    :return: A dictionary containing weights for each class
    '''
    weight_neg = 0.5 * (num_neg + num_pos) / (num_neg)
    weight_pos = 0.5 * (num_neg + num_pos) / (num_pos)
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
    ros = RandomOverSampler(random_state=0)
    X_resampled, Y_resampled = ros.fit_resample(X_train, Y_train)
    print("Train set shape before oversampling: ", X_train.shape, " Train set shape after resampling: ", X_resampled.shape)
    return X_resampled, Y_resampled

def train_model(save_weights=True, write_logs=True):
    '''
    Defines and trains HIFIS-v2 model. Prints and logs relevant metrics.
    :param save_weights: A flag indicating whether to save the model weights
    :return: A dictionary of metrics on the test set
    '''

    # Load project config data
    input_stream = open(os.getcwd() + "/config.yml", 'r')
    cfg = yaml.full_load(input_stream)

    # Load config data generated from preprocessing
    input_stream = open(os.getcwd() + cfg['PATHS']['INTERPRETABILITY'], 'r')
    cfg_gen = yaml.full_load(input_stream)
    noncat_features = cfg_gen['NON_CAT_FEATURES']   # Noncategorical features to be scaled
    plot_path = cfg['PATHS']['IMAGES']  # Path for images of matplotlib figures

    # Load and partition dataset
    df_ohe = pd.read_csv(cfg['PATHS']['PROCESSED_OHE_DATA'])
    df = pd.read_csv(cfg['PATHS']['PROCESSED_DATA'])   # Data prior to one hot encoding
    num_neg, num_pos = np.bincount(df_ohe['GroundTruth'])
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
    Y_train = np.array(train_df_ohe.pop('GroundTruth'))
    Y_val = np.array(val_df_ohe.pop('GroundTruth'))
    Y_test = np.array(test_df_ohe.pop('GroundTruth'))

    # Convert dataframes to numpy arrays
    X_train = np.array(train_df_ohe)
    X_val = np.array(val_df_ohe)
    X_test = np.array(test_df_ohe)

    # Normalize numerical data
    col_trans_scaler = ColumnTransformer(transformers=[('col_trans_ordinal', StandardScaler(), noncat_feat_idxs)],
                                         remainder='passthrough')
    X_train = col_trans_scaler.fit_transform(X_train)
    X_val = col_trans_scaler.transform(X_val)
    X_test = col_trans_scaler.transform(X_test)
    dump(col_trans_scaler, cfg['PATHS']['SCALER_COL_TRANSFORMER'], compress=True)

    # Define metrics.
    metrics = [BinaryAccuracy(name='accuracy'), Precision(name='precision'), Recall(name='recall'), AUC(name='auc'),
               F1Score(name='f1')]

    # Set callbacks.
    early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=15, mode='min', restore_best_weights=True)
    callbacks = [early_stopping]
    if write_logs:
        log_dir = cfg['PATHS']['LOGS'] + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard)

    # Define the model.
    model = model1(cfg['NN']['MODEL1'], (X_train.shape[-1],), metrics)   # Build model graph

    # Calculate class weights
    class_weight = None
    if cfg['TRAIN']['CLASS_WEIGHT']:
        class_weight = get_class_weights(num_pos, num_neg)

    # Oversample minority class
    if cfg['TRAIN']['MINORITY_OVERSAMPLE']:
        X_train, Y_train = minority_oversample(X_train, Y_train)

    # Train model for a specified number of times and keep the one with the best target test metric
    metric_monitor = cfg['TRAIN']['METRIC_MONITOR']
    best_value = 1000.0 if metric_monitor == 'loss' else 0.0
    for i in range(cfg['TRAIN']['NUM_RUNS']):
        print("Training run ", i+1, " / ", cfg['TRAIN']['NUM_RUNS'])

        # Train the model.
        history = model.fit(X_train, Y_train, batch_size=cfg['TRAIN']['BATCH_SIZE'], epochs=cfg['TRAIN']['EPOCHS'],
                          validation_data=(X_val, Y_val), callbacks=callbacks, class_weight=class_weight)

        # Run the model on the test set and print the resulting performance metrics.
        test_results = model.evaluate(X_test, Y_test)
        test_metrics = {}
        test_summary_str = [['**Metric**','**Value**']]
        for metric, value in zip(model.metrics_names, test_results):
            test_metrics[metric] = value
            print(metric, ' = ', value)
            test_summary_str.append([metric, str(value)])

        # If this model outperforms the previous ones based on the specified metric, save this one.
        if (((metric_monitor == 'loss') and (test_metrics[metric_monitor] < best_value))
                or ((metric_monitor != 'loss') and (test_metrics[metric_monitor] > best_value))):
            best_value = test_metrics[metric_monitor]
            best_model = model
            best_metrics = test_metrics
    print("Best model test metrics: ", best_metrics)

    # Visualize metrics about the training process
    test_predictions = best_model.predict(X_test, batch_size=cfg['TRAIN']['BATCH_SIZE'])
    metrics_to_plot = ['loss', 'auc', 'precision', 'recall', 'f1']
    plot_metrics(history, metrics_to_plot, file_path=plot_path)
    roc_img = plot_roc("Test set", Y_test, test_predictions, file_path=None)
    cm_img = plot_confusion_matrix(Y_test, test_predictions, file_path=None)

    # Log test set results and plots in TensorBoard
    if write_logs:
        writer = tf.summary.create_file_writer(logdir=log_dir)
        with writer.as_default():
            tf.summary.text(name='Test set metrics', data=tf.convert_to_tensor(test_summary_str), step=0)
            tf.summary.image(name='ROC Curve (Test Set)', data=roc_img, step=0)
            tf.summary.image(name='Confusion Matrix (Test Set)', data=cm_img, step=0)

    if save_weights:
        save_model(best_model, cfg['PATHS']['MODEL_WEIGHTS'])        # Save model weights
    return test_metrics

if __name__ == '__main__':
    results = train_model(save_weights=True, write_logs=True)

