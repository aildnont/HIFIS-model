from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Reshape, concatenate, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import Constant
from tensorflow import convert_to_tensor, split, reshape, transpose
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, log_loss
from xgboost import XGBClassifier
import numpy as np
from src.custom.losses import f1_loss

def hifis_mlp(cfg, input_dim=None, metrics=None, metadata=None, output_bias=None, hparams=None):
    '''
    Defines a Keras model for HIFIS multi-layer perceptron model (i.e. HIFIS-v2)
    :param cfg: A dictionary of parameters associated with the model architecture
    :param input_dim: The shape of the model input
    :param metrics: Metrics to track model's performance
    :param metadata: Dict containing prediction horizon
    :param output_bias: initial bias applied to output layer
    :param hparams: dict of hyperparameters
    :return: a Keras model object with the architecture defined in this method
    '''

    # Set hyperparameters
    if hparams is None:
        nodes_dense0 = cfg['NODES']['DENSE0']
        nodes_dense1 = cfg['NODES']['DENSE1']
        layers = cfg['LAYERS']
        dropout = cfg['DROPOUT']
        l2_lambda = cfg['L2_LAMBDA']
        lr = cfg['LR']
        optimizer = Adam(learning_rate=lr)
    else:
        nodes_dense0 = hparams['NODES0']
        nodes_dense1 = hparams['NODES1']
        layers = hparams['LAYERS']
        dropout = hparams['DROPOUT']
        lr = 10 ** hparams['LR']    # Random sampling on logarithmic scale
        beta_1 = 1 - 10 ** hparams['BETA_1']
        beta_2 = 1 - 10 ** hparams['BETA_2']
        l2_lambda = 10 ** hparams['L2_LAMBDA']
        if hparams['OPTIMIZER'] == 'adam':
            optimizer = Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2)
        elif hparams['OPTIMIZER'] == 'sgd':
            optimizer = SGD(learning_rate=lr)

    if output_bias is not None:
        output_bias = Constant(output_bias)

    # Define model architecture.
    model = Sequential(name='HIFIS-mlp_' + str(metadata['N_WEEKS']) + '-weeks')
    model.add(Dense(nodes_dense0, input_shape=input_dim, activation='relu', kernel_regularizer=l2(l2_lambda),
              bias_regularizer=l2(l2_lambda), name="dense0"))
    model.add(Dropout(dropout, name='dropout0'))
    for i in range(1, layers):
        model.add(Dense(nodes_dense1, activation='relu', kernel_regularizer=l2(l2_lambda),
                  bias_regularizer=l2(l2_lambda), name='dense%d'%i))
        model.add(Dropout(dropout, name='dropout%d'%i))
    model.add(Dense(1, activation='sigmoid', name="output", bias_initializer=output_bias))

    # Set model loss function, optimizer, metrics.
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)

    # Print summary of model and return model object
    if hparams is None:
        model.summary()
    return model


def hifis_rnn_mlp(cfg, input_dim=None, metrics=None, metadata=None, output_bias=None, hparams=None):
    '''
    Defines a Keras model for HIFIS hybrid recurrent neural network and multilayer perceptron model (i.e. HIFIS-v3)
    :param cfg: A dictionary of parameters associated with the model architecture
    :param input_dim: The shape of the model input
    :param metrics: Metrics to track model's performance
    :param metadata: Dict containing prediction horizon, time series feature info
    :param output_bias: initial bias applied to output layer
    :param hparams: dict of hyperparameters
    :return: a Keras model object with the architecture defined in this method
    '''

    # Set hyperparameters
    if hparams is None:
        nodes_dense0 = cfg['DENSE']['DENSE0']
        nodes_dense1 = cfg['DENSE']['DENSE1']
        layers = cfg['LAYERS']
        dropout = cfg['DROPOUT']
        l2_lambda = cfg['L2_LAMBDA']
        lr = cfg['LR']
        optimizer = Adam(learning_rate=lr)
        lstm_units = cfg['LSTM']['UNITS']
    else:
        nodes_dense0 = hparams['NODES0']
        nodes_dense1 = hparams['NODES1']
        layers = hparams['LAYERS']
        dropout = hparams['DROPOUT']
        lr = 10 ** hparams['LR']    # Random sampling on logarithmic scale
        beta_1 = 1 - 10 ** hparams['BETA_1']
        beta_2 = 1 - 10 ** hparams['BETA_2']
        l2_lambda = 10 ** hparams['L2_LAMBDA']
        lstm_units = hparams['LSTM_UNITS']

        if hparams['OPTIMIZER'] == 'adam':
            optimizer = Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2)
        elif hparams['OPTIMIZER'] == 'sgd':
            optimizer = SGD(learning_rate=lr)

    if output_bias is not None:
        output_bias = Constant(output_bias)

    # Receive input to model and split into 2 tensors containing dynamic and static features respectively
    X_input = Input(shape=input_dim)
    split_idx = metadata['NUM_TS_FEATS'] * metadata['T_X']
    X_dynamic, X_static = split(X_input, [split_idx, X_input.shape[1] - split_idx], axis=1)

    # Define RNN component of model using LSTM cells. LSTM input shape is [batch_size, timesteps, features]
    lstm_input_shape = (metadata['T_X'], metadata['NUM_TS_FEATS'])
    X_dynamic = Reshape(lstm_input_shape)(X_dynamic)
    X_dynamic = LSTM(lstm_units, activation='tanh', return_sequences=True)(X_dynamic)
    X_dynamic = Flatten()(X_dynamic)

    # Define MLP component of model
    X = concatenate([X_dynamic, X_static])      # Combine output of LSTM with static features
    X = Dense(nodes_dense0, input_shape=input_dim, activation='relu', kernel_regularizer=l2(l2_lambda),
              bias_regularizer=l2(l2_lambda), name="dense0")(X)
    X = Dropout(dropout, name='dropout0')(X)
    for i in range(1, layers):
        X = Dense(nodes_dense1, activation='relu', kernel_regularizer=l2(l2_lambda),
                  bias_regularizer=l2(l2_lambda), name='dense%d'%i)(X)
        X = Dropout(dropout, name='dropout%d'%i)(X)
    Y = Dense(1, activation='sigmoid', name="output", bias_initializer=output_bias)(X)

    # Define model with inputs and outputs
    model = Model(inputs=X_input, outputs=Y, name='HIFIS-rnn-mlp_' + str(metadata['N_WEEKS']) + '-weeks')

    # Set model loss function, optimizer, metrics.
    model.compile(loss=f1_loss(4.5), optimizer=optimizer, metrics=metrics)

    # Print summary of model and return model object
    if hparams is None:
        model.summary()
    return model


class logistic_regression:

    def __init__(self, cfg, metrics=[], **kwargs):
        self.cfg = cfg
        self.metrics = [None] + [m for m in metrics if not isinstance(m, str)]
        self.metrics_names = ['loss'] + [m.name for m in metrics if not isinstance(m, str)]

    def fit(self, X_train, Y_train, class_weight=None, **kwargs):
        self.model = LogisticRegression(class_weight=class_weight)
        self.model.fit(X_train, Y_train)

    def evaluate(self, X_test, Y_test):
        test_preds = self.model.predict(X_test)
        test_probs = self.model.predict_proba(X_test)[:, 1]
        test_metrics = []
        for i in range(len(self.metrics_names)):
            if self.metrics_names[i] in ['precision', 'recall', 'f1score']:
                scores = []
                for t in self.metrics[i].thresholds:
                    preds = (test_probs >= t).astype(int)
                    if self.metrics_names[i] == 'precision':
                        scores.append(precision_score(Y_test, preds))
                    elif self.metrics_names[i] == 'recall':
                        scores.append(recall_score(Y_test, preds))
                    elif self.metrics_names[i] == 'f1score':
                        scores.append(f1_score(Y_test, preds))
                if len(scores) == 1:
                    scores = scores[0]
                test_metrics.append(np.array(scores))
            elif self.metrics_names[i] == 'auc':
                test_metrics.append(roc_auc_score(Y_test, test_probs))
            elif self.metrics_names[i] == 'loss':
                test_metrics.append(log_loss(Y_test, test_probs))
            elif self.metrics_names[i] == 'accuracy':
                test_metrics.append(accuracy_score(Y_test, test_preds))
        return test_metrics


class random_forest:

    def __init__(self, cfg, metrics=[], **kwargs):
        self.cfg = cfg
        self.metrics = [None] + [m for m in metrics if not isinstance(m, str)]
        self.metrics_names = ['loss'] + [m.name for m in metrics if not isinstance(m, str)]

    def fit(self, X_train, Y_train, class_weight=None, **kwargs):
        self.model = RandomForestClassifier(n_estimators=self.cfg['N_ESTIMATORS'], class_weight=class_weight)
        self.model.fit(X_train, Y_train)

    def evaluate(self, X_test, Y_test):
        test_preds = self.model.predict(X_test)
        test_probs = self.model.predict_proba(X_test)[:, 1]
        test_metrics = []
        for i in range(len(self.metrics_names)):
            if self.metrics_names[i] in ['precision', 'recall', 'f1score']:
                scores = []
                for t in self.metrics[i].thresholds:
                    preds = (test_probs >= t).astype(int)
                    if self.metrics_names[i] == 'precision':
                        scores.append(precision_score(Y_test, preds))
                    elif self.metrics_names[i] == 'recall':
                        scores.append(recall_score(Y_test, preds))
                    elif self.metrics_names[i] == 'f1score':
                        scores.append(f1_score(Y_test, preds))
                if len(scores) == 1:
                    scores = scores[0]
                test_metrics.append(np.array(scores))
            elif self.metrics_names[i] == 'auc':
                test_metrics.append(roc_auc_score(Y_test, test_probs))
            elif self.metrics_names[i] == 'loss':
                test_metrics.append(log_loss(Y_test, test_probs))
            elif self.metrics_names[i] == 'accuracy':
                test_metrics.append(accuracy_score(Y_test, test_preds))
        return test_metrics


class xgboost_model:

    def __init__(self, cfg, metrics=[], **kwargs):
        self.cfg = cfg
        self.metrics = [None] + [m for m in metrics if not isinstance(m, str)]
        self.metrics_names = ['loss'] + [m.name for m in metrics if not isinstance(m, str)]

    def fit(self, X_train, Y_train, class_weight=None, **kwargs):
        if class_weight is not None:
            scale_pos_weight = class_weight[1] / class_weight[0]
        else:
            scale_pos_weight = None
        self.model = XGBClassifier(n_estimators=self.cfg['N_ESTIMATORS'], scale_pos_weight=scale_pos_weight)
        self.model.fit(X_train, Y_train)

    def evaluate(self, X_test, Y_test):
        test_preds = self.model.predict(X_test)
        test_probs = self.model.predict_proba(X_test)[:, 1]
        test_metrics = []
        for i in range(len(self.metrics_names)):
            if self.metrics_names[i] in ['precision', 'recall', 'f1score']:
                scores = []
                for t in self.metrics[i].thresholds:
                    preds = (test_probs >= t).astype(int)
                    if self.metrics_names[i] == 'precision':
                        scores.append(precision_score(Y_test, preds))
                    elif self.metrics_names[i] == 'recall':
                        scores.append(recall_score(Y_test, preds))
                    elif self.metrics_names[i] == 'f1score':
                        scores.append(f1_score(Y_test, preds))
                if len(scores) == 1:
                    scores = scores[0]
                test_metrics.append(np.array(scores))
            elif self.metrics_names[i] == 'auc':
                test_metrics.append(roc_auc_score(Y_test, test_probs))
            elif self.metrics_names[i] == 'loss':
                test_metrics.append(log_loss(Y_test, test_probs))
            elif self.metrics_names[i] == 'accuracy':
                test_metrics.append(accuracy_score(Y_test, test_preds))
        return test_metrics