from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Reshape, concatenate, Flatten, GRU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import Constant
from tensorflow import convert_to_tensor, split, reshape, transpose
from src.custom.losses import f1_loss

def hifis_mlp(cfg, input_dim, metrics, metadata, output_bias=None, hparams=None):
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


def hifis_rnn_mlp(cfg, input_dim, metrics, metadata, output_bias=None, hparams=None):
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
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)

    # Print summary of model and return model object
    if hparams is None:
        model.summary()
    return model


def hifis_systemwide_rnn(cfg, input_dim, metrics, metadata, output_bias=None, hparams=None):
    '''
    Defines a Keras model for HIFIS hybrid recurrent neural network and multilayer perceptron model (i.e. HIFIS-v3)
    :param cfg: A dictionary of parameters associated with the model architecture
    :param input_dim: The shape of the model input
    :param metrics: Metrics to track model's performance
    :param metadata: Dict containing prediction horizon, time series feature info
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

    # Define RNN component using LSTM cells. LSTM input shape is [batch_size, timesteps, features]
    lstm_input_shape = (metadata['T_X'] - 1, metadata['NUM_TS_FEATS'])
    X = Reshape(lstm_input_shape)(X_input)
    X = LSTM(lstm_units, activation='tanh', return_sequences=True)(X)
    X = Flatten()(X)
    X = Dropout(dropout)(X)
    X = Dense(nodes_dense0, activation='relu', kernel_regularizer=l2(l2_lambda),
                    bias_regularizer=l2(l2_lambda), name='dense0')(X)
    for i in range(1, layers):
        X = Dense(nodes_dense1, activation='relu', kernel_regularizer=l2(l2_lambda),
                  bias_regularizer=l2(l2_lambda), name='dense%d'%i)(X)
    Y = Dense(metadata['NUM_TS_FEATS'], activation='relu', name="output", bias_initializer=output_bias)(X)

    # Define model with inputs and outputs
    model = Model(inputs=X_input, outputs=Y, name='HIFIS-systemwide-rnn_' + str(metadata['N_WEEKS']) + '-weeks')

    # Set model loss function, optimizer, metrics.
    model.compile(loss='mae', optimizer=optimizer, metrics=metrics)

    # Print summary of model and return model object
    if hparams is None:
        model.summary()
    return model
