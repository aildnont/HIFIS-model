import os

import pytest
import yaml
import numpy as np
import tensorflow as tf

from src.models.models import hifis_rnn_mlp

@pytest.fixture
def cfg():
    return yaml.full_load(open(os.getcwd() + "/tests/hifis_rnn_mlp_test.yml", 'r'))

@pytest.fixture
def model(cfg):
    metadata = {
        'T_X': cfg['T_X'],
        'NUM_TS_FEATS': cfg['TIME_SERIES_FEATS'],
        'N_WEEKS': cfg['N_WEEKS']
    }
    return hifis_rnn_mlp(cfg, input_dim=(cfg['TOTAL_FEATS']), metrics=None, metadata=metadata, output_bias=None)

@pytest.fixture
def input_data(cfg):
    T_x = cfg['T_X']
    n_ts_feats = cfg['TIME_SERIES_FEATS']
    n_dynamic_inputs = n_ts_feats * T_x
    n_static_inputs = cfg['TOTAL_FEATS'] - n_dynamic_inputs

    # Repeat [1, 2, ..., n_static_inputs] for each time step
    dynamic_inputs = np.concatenate([
        np.expand_dims(np.arange(1,n_ts_feats + 1), 0) for i in range(T_x)
    ], axis=1)

    # Concatenate static and dynamic features
    test_input = np.concatenate([np.zeros((1, n_static_inputs)), dynamic_inputs], axis=1)
    print(f"Test input: {test_input}")
    return test_input

def test_hifis_rnn_mlp_architecture(cfg, model, input_data):

    layer_outputs = [layer.output for layer in model.layers]
    intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    intermediate_outputs = intermediate_model.predict(input_data)

    T_x = cfg['T_X']
    n_ts_feats = cfg['TIME_SERIES_FEATS']
    n_dynamic_inputs = n_ts_feats * T_x
    n_total_feats = cfg['TOTAL_FEATS']

    reshape_feats = intermediate_outputs[1]     # Intermediate output after splitting dynamic and static features
    static_feats = reshape_feats[0]
    assert np.array_equal(static_feats, np.zeros((1,n_total_feats-n_dynamic_inputs))), \
        "Static component of input contains time series features and is of correct shape"

    dynamic_feats = reshape_feats[1]
    assert all([dynamic_feats[0, i] > 0 for i in range(n_dynamic_inputs)]), "Dynamic component of input contains static features"
    assert dynamic_feats.shape[1] == n_dynamic_inputs, "Incorrect number of dynamic features"

    reshaped_dynamic_feats = np.reshape(dynamic_feats, (T_x, n_ts_feats))
    assert all([np.array_equal(reshaped_dynamic_feats[i], np.arange(1, n_ts_feats + 1)) for i in range(T_x)]), \
        "Dynamic component of input is incorrectly formatted"

if __name__ == "__main__":
    pytest.main([__file__])