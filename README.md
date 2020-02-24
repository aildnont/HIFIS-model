# HIFIS-v2 Model
![alt text](documents/readme_images/london_logo.png "A sample LIME
explanation") ![alt text](documents/readme_images/hifis_logo.png "A
sample LIME explanation")

The purpose of this project is to investigate the efficacy of a machine
learning solution to assist in identifying individuals at risk of
chronic homelessness. A model prototype was built for the municipal
government of the City of London, Ontario, Canada. This repository
contains the code used to train a neural network model to classify
clients in the city's [Homeless Individuals and Families Information
System](https://www.canada.ca/en/employment-social-development/programs/homelessness/hifis.html)
(HIFIS) database as either at risk or not at risk of chronic
homelessness within a specified predictive horizon. In an effort to
anticipate forthcoming federal and provincial regulation of automated
decision-making systems, this repository applies interpretability and
bias-reducing methods to explain the model's predictions. This
repository is intended to serve as a turnkey template for other
municipalities using the HIFIS database schema who wish to explore the
application of this model in their own locales.

## Getting Started
1. Clone this repository (for help see this
   [tutorial](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)).
2. Install the necessary dependencies (listed in
   [requirements.txt](requirements.txt)). To do this, open a terminal in
   the root directory of the project and run the following:
   ```
   $ pip install -r requirements.txt
   ```
3. Open [_export_clients_to_csv.ps1_](export_clients_to_csv.ps1) for
   editing. Replace "_[Instance Name goes here]_" with your HIFIS
   database instance name. Execute
   [_export_clients_to_csv.ps1_](export_clients_to_csv.ps1). A file
   named _"HIFIS_Clients.csv"_ should now be within the _data/raw/_
   folder. See
   [HIFIS_Clients_example.csv](data/raw/HIFIS_Clients_example.csv) for
   an example of the column names in our _"HIFIS_Clients.csv"_ (note
   that the data is fabricated; this file is included for illustrative
   purposes).
4. Check that your features in _HIFIS_Clients.csv_ match in
   [config.yml](config.yml). If necessary, update feature
   classifications in this file (for help see
   [Project Config](#project-config)).
5. Execute [_preprocess.py_](src/data/preprocess.py) to transform the
   data into the format required by the machine learning model.
   Preprocessed data will be saved within _data/preprocessed/_.
6. Execute [_train.py_](src/train.py) to train the neural network model
   on your preprocessed data. The trained model weights will be saved
   within _results/models/_, and its filename will resemble the
   following structure: modelyyyymmdd-hhmmss.h5, where yyyymmdd-hhmmss
   is the current time. The
   [TensorBoard](https://www.tensorflow.org/tensorboard) log files will
   be saved within _results/logs/training/_.
7. In [config.yml](config.yml), set _MODEL_TO_LOAD_ within _PATHS_ to
   the path of the model weights file that was generated in step 6 (for help see
   [Project Config](#project-config)).
   Execute [_lime_explain.py_](src/interpretability/lime_explain.py) to
   generate interpretable explanations for the model's predictions on
   the test set. A spreadsheet of predictions and explanations will be
   saved within _results/experiments/_.

## Use Cases

### Train a model and visualize results
1. Once you have _HIFIS_Clients.csv_ sitting in the raw data folder
   (_data/raw/), execute [_preprocess.py_](src/data/preprocess.py). See
   [Getting Started](#getting-started) for help obtaining
   _HIFIS_Clients.csv_.
2. Ensure data has been preprocessed properly. That is, verify that
   _data/processed/_ contains both _HIFIS_Processed.csv_ and
   _HIFIS_Processed_OHE.csv_. The latter is identical to the former with
   the exception being that its single-valued categorical features have
   been one-hot encoded.
3. In [config.yml](config.yml), set _EXPERIMENT_TYPE_ within _TRAIN_ to
   _'single_train'_.
4. Execute [train.py](src/train.py). The trained model's weights will be
   located in _results/models/_, and its filename will resemble the
   following structure: modelyyyymmdd-hhmmss.h5, where yyyymmdd-hhmmss
   is the current time. The model's logs will be located in
   _results/logs/training/_, and its directory name will be the current
   time in the same format. These logs contain information about the
   experiment, such as metrics throughout the training process on the
   training and validation sets, and performance on the test set. The
   logs can be visualized by running
   [TensorBoard](https://www.tensorflow.org/tensorboard) locally. See
   below for an example of a plot from a TensorBoard log file depicting
   loss on the training and validation sets vs. epoch. Plots depicting
   the change in performance metrics throughout the training process
   (such as the example below) are available in the _SCALARS_ tab of
   TensorBoard.  
   ![alt text](documents/readme_images/tensorboard_loss.png "Loss vs
   Epoch")  
   You can also visualize the trained model's performance on the test
   set. See below for an example of the ROC Curve and Confusion Matrix
   based on test set predictions. In our implementation, these plots are
   available in the _IMAGES_ tab of TensorBoard.  
   ![alt text](documents/readme_images/roc_example.png "ROC Curve")
   ![alt text](documents/readme_images/cm_example.png "Confusion
   Matrix")

### Train multiple models and save the best one
Not every model trained will perform at the same level on the test set.
This procedure enables you to train multiple models and save the one
that scored the best result on the test set for a particular metric that
you care about optimizing.
1. Follow steps 1 and 2 in
   [Train a model and visualize results](#train-a-model-and-visualize-results).
2. In [config.yml](config.yml), set _EXPERIMENT_TYPE_ within _TRAIN_ to
   _'multi_train'_.
3. Decide which metric you would like to optimize. In
   [config.yml](config.yml), set _METRIC_MONITOR_ within _TRAIN_ to your
   chosen metric. For example, if you decide to select the model with
   the best recall on the test set, set this field to _'recall'_.
4. Decide how many models you wish to train. In
   [config.yml](config.yml), set _NUM_RUNS_ within _TRAIN_ to your
   chosen number of training sessions. For example, if you wish to train
   10 models, set this field to _10_.
5. Execute [train.py](src/train.py). The weights of the model that had
   the best performance on the test set for the metric you specified
   will be located in _results/models/training/_, and its filename will
   resemble the following structure: modelyyyymmdd-hhmmss.h5, where
   yyyymmdd-hhmmss is the current time. The model's logs will be located in
   _results/logs/training/_, and its directory name will be the current
   time in the same format.

### Prediction Horizon Search Experiment
The prediction horizon (_N_) is defined as the amount of time from now
that the model makes its predictions for. In our case, the prediction
horizon is how far in the future (in weeks) the model is predicting risk
of chronic homelessness. For example, if the _N_ = 26 weeks, then the
model is predicting whether or not a client will be at risk of chronic
homelessness in 26 weeks. While developing this model, we noticed that
the model's performance is inversely correlated with the prediction
horizon. The Prediction Horizon Search Experiment trains the model
multiple times at multiple values of _N_. For each value of _N_, the
data is retrospectively preprocessed by cutting off the most recent _N_
weeks of records. The relationships of _N_ and several model metrics are
graphed for the user to deliver insight on the impact of _N_ and make a
business decision as to which value yields optimal results. See below
for instructions on how to run a Prediction Horizon Search Experiment.
1. In the _HORIZON_SEARCH_ section of [config.yml](config.yml), set
   _N_MIN_, _N_MAX_, _N_INTERVAL_ and _RUNS_PER_N_ according to your
   organization's needs (see [Project Config](#project-config) for help).
2. Run _src/horizon_search.py_. This may take several minutes to hours,
   depending on your hardware and settings from the previous step.
3. A .csv representation of experiment results will be available within
   _results/experiments/_, called _horizon_searchyyyymmdd-hhmmss.csv_,
   where yyyymmdd-hhmmss is the current time. A graphical representation
   of the results will be available within
   _documents/generated_images/_, called
   _horizon_experiment_yyyymmdd-hhmmss.png_. See below for an example of
   this visualization.

![alt text](documents/readme_images/horizon_experiment_example.png
"Prediction Horizon Search Experiment")

### LIME Explanations
Since the predictions made by this model are to be used by a government
institution to benefit vulnerable members of society, it is imperative
that the model's predictions may be explained so as to facilitate
ensuring the model is making responsible predictions, as well as
assuring transparency and accountability of government decision-making
processes. Since this model is a neural network, it is difficult to
decipher which rules or heuristics it is employing to make its
predictions. Interpretability in machine learning is a growing concern,
especially with applications in the healthcare and social services
domains. We used [Local Interpretable
Model-Agnostic Explanations](https://arxiv.org/pdf/1602.04938.pdf) (i.e.
LIME) to explain the predictions of the neural network classifier that
we trained. We used the implementation available in the authors' [GitHub
repository](https://github.com/marcotcr/lime). LIME perturbs the
features in an example and fits a linear model to approximate the neural
network at the local region in the feature space surrounding the
example. It then uses the linear model to determine which features were
most contributory to the model's prediction for that example. By
applying LIME to our trained model, we can conduct informed feature
engineering based on any obviously inconsequential features we see (e.g.
_EyeColour_) or insights from domain experts. We can also tell if the
model is learning any unintended bias and eliminate that bias through
additional feature engineering. See the steps below to apply LIME to
explain the model's predictions on examples in the test set.
1. Having previously run _[train.py](src/train.py)_, ensure that
   _data/processed/_ contains both _Train_Set.csv_ and _Test_Set.csv_.
2. In [config.yml](config.yml), set _MODEL_TO_LOAD_ within _PATHS_ to
   the path of the model weights file (_.h5_ file) that you wish to use
   for prediction.
3. In the _main_ function of
   _[lime_explain.py](src/interpretability/lime_explain.py)_, you can
   select to either perform a LIME experiment or run LIME on 1 test set
   example. Uncomment the function you wish to execute.
   1. You can call `run_lime_experiment_and_visualize(lime_dict)`, which
      will run LIME on all examples in the test set, create a .csv file
      of the results, and produce a visualization of the average
      explainable feature rules. The .csv file will be located in
      _results/experiments/_, and will be called
      _lime_experimentyyyymmdd-hhmmss.csv_, where yyyymmdd-hhmmss is the
      current time. The visualization will be located in
      _documents/generated_images/_, and will be called
      _LIME_Eplanations_yyyymmdd-hhmmss.csv_.
   2. You can call `explain_single_client(lime_dict, client_id)`, which
      will run LIME on the example in the test set whose ClientID is
      that which you passed to the function. A graphic will be generated
      that depicts the top explainable features that the model used to
      make its prediction. The graphic will be displayed in a new window
      for the user, but it will not be automatically saved. See below
      for an example of this graphic.
4. Interpret the output of the LIME explainer. LIME partitions features
   into classes or ranges and reports the features most contributory to
   a prediction. A feature explanation is considered to be a value (or
   range of values) of a feature and its associated weight in the
   prediction. In the example portrayed by the bar graph below, the fact
   that _TotalStays_ was greater than 4 but less than or equal to 23
   contributed negatively with a magnitude of about 0.22 to a positive
   prediction (meaning it contributed at a magnitude of 0.22 toward a
   negative prediction). As another example, the rule
   _"ReasonForService_Streets=1"_ indicates that at some point the client
   has a record that cites their reason for service as _"Streets"_ (_=1_
   indicates that a Boolean feature is present, and _=0_ indicates that
   a Boolean feature is not present) and that this explanation
   contributed with a weight of about 0.02 toward a positive prediction.
   As one last example, consider that this client's _AboriginalIndicator_
   value is _"Yes - Tribe Not Known"_, which contributed with a weight of
   about 0.04 towards a negative prediction.
   
**NOTE**: Many clients have incomplete records. To represent missing values, default values are inserted into the dataset. You may see these values when examining LIME explanations.
- Missing records for numerical features are given a value of _-1_
- Missing records for categorical features are given a value of _"Unknown"_

![alt text](documents/readme_images/LIME_example.PNG "A sample LIME
explanation")

### Random Hyperparameter Search
Hyperparameter tuning is an important part of the standard machine
learning workflow. We chose to conduct a series of random hyperparameter
searches. The results of one search informed the next, leading us to
eventually settle on the hyperparameters currently set in the _TRAIN_
and _NN_ sections of [config.yml](config.yml). We applied TensorBoard
visualization to aid the random hyperparameter search. With the help of
the [HParam
Dashboard](https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams),
one can see the effect of different combinations of hyperparameters on
the model's test set performance metrics.

In our random hyperparameter search, we study the effects of _x_ random
combinations of hyperparameters by training the model _y_ times for each
of the _x_ combinations and recording the results. See the steps below
on how to conduct a random hyperparameter search with our implementation
for the following 3 hyperparameters: _dropout_, _learning rate_, and
_layers_.
1. In the in the _HP_ subsection of the _TRAIN_ section of
   [config.yml](config.yml), set the number of random combinations of
   hyperparameters you wish to study and the number of times you would
   like to train the model for each combination (see
   [Project Config](#train) for help).
   ```
   COMBINATIONS: 60
   REPEATS: 2
   ```
2. Set the ranges of hyperparameters you wish to study in the _HP_
   subsection of the _TRAIN_ section of [config.yml](config.yml).
   Consider whether your hyperparameter ranges are continuous (i.e.
   real) or discrete and whether any need to be investigated on the
   logarithmic scale.
   ```
   DROPOUT: [0.2, 0.5]          # Real range
   LR: [-4.0, -2.5]             # Real range on logarithmic scale (10^x)   
   LAYERS: [2, 3, 4]            # Discrete range
   ```
3.  Within the _random_hparam_search()_ function defined in
    [train.py](src/train.py), add your hyperparameters as HParam objects
    to the list of hyperparameters being considered.
    ```
    HPARAMS.append(hp.HParam('DROPOUT', hp.RealInterval(hp_ranges['DROPOUT'][0], hp_ranges['DROPOUT'][1])))
    HPARAMS.append(hp.HParam('LR', hp.RealInterval(hp_ranges['LR'][0], hp_ranges['LR'][1])))
    HPARAMS.append(hp.HParam('LAYERS', hp.Discrete(hp_ranges['LAYERS'])))
    ```
4. In the appropriate location (varies by hyperparameter), ensure that
   you set the hyperparameters based on the random combination. In our
   example, all of these hyperparameters are set in the model definition
   (i.e. within _model1()_ in [model.py](src/models/models.py)). You may
   have to search the code to determine where to set your particular
   choice of hyperparameters.
   ```
   dropout = hparams['DROPOUT']
   lr = 10 ** hparams['LR']             # Transform to logarithmic scale
   layers = hparams['LAYERS']
   ```
5.  In [config.yml](config.yml), set _EXPERIMENT_TYPE_ within the
    _TRAIN_ section to _'hparam_search'_.
6. Execute [train.py](src/train.py). The experiment's logs will be
   located in _results/logs/hparam_search/_, and the directory name will
   be the current time in the following format: _yyyymmdd-hhmmss_. These
   logs contain information on test set metrics with models trained on
   different combinations of hyperparameters. The logs can be visualized
   by running [TensorBoard](https://www.tensorflow.org/tensorboard)
   locally. See below for an example of a view offered by the HParams
   dashboard of TensorBoard. Each point represents 1 training run. The
   graph compares values of hyperparameters to test set metrics.

![alt text](documents/readme_images/hparam_example.png "A sample HParams
dashboard view")

### Bulk predictions from raw data
Once a trained model is produced, the user may wish to obtain
predictions and explanations for all clients currently in the HIFIS
database. As clients' life situations change over time, their records in
the HIFIS database change as well. Thus, it is useful to rerun
predictions for clients every so often. If you wish to track changes in
predictions and explanations for particular clients over time, you can
choose to append timestamped predictions to a file containing previous
timestamped predictions. The steps below detail how to run prediction
for all clients, given raw data from HIFIS and a trained model.
1. Ensure that you have already run
   _[lime_explain.py](src/interpretability/lime_explain.py)_ after
   training your model, as it will have generated and saved a
   LIME Explainer object at _data/interpretability/lime_explainer.pkl_.
2. Ensure that you have _HIFIS_Clients.csv_ located within in the raw
   data folder (_data/raw/_). See [Getting Started](#getting-started)
   for help obtaining _HIFIS_Clients.csv_.
3. In [config.yml](config.yml), set _MODEL_TO_LOAD_ within _PATHS_ to
   the path of the model weights file (_.h5_ file) that you wish to use
   for prediction.
4. In the _main_ function of _[predict.py](src/predict.py)_, you can opt
   to either save predictions to a new file or append predictions and
   their corresponding timestamps to a file containing past predictions.
   Ensure the function you wish to execute is uncommented.
   1. You can call `results = predict_and_explain_set(data_path=None,
      save_results=True, give_explanations=True)`, which will preprocess
      raw client data, run prediction for all clients, and run LIME to
      explain these predictions. Results will be saved in a .csv file,
      which will be located in _results/predictions/_, and will be
      called _predictionsyyyymmdd-hhmmss.csv_, where yyyymmdd-hhmmss is
      the current time.
   2. You can call `trending_prediction(data_path=None)`, which will
      produce predictions and explanations in the same method as
      described in (i), but will include timestamps for when the
      predictions were made. The results will be appended to a file
      called _trending_predictions.csv_, located within
      _results/prediction/_. This file contains predictions made at
      previous times, enabling the user to compare the change in
      predictions and explanations for particular clients over time.

## Project Structure
The project looks similar to the directory structure below. Disregard
any _.gitkeep_ files, as their only purpose is to force Git to track
empty directories.

```
├── data
│   ├── interpretability          <- Generated feature information
│   ├── processed                 <- Products of preprocessing
│   ├── raw                       <- Raw data from SQL query
│   └── transformers              <- Serialized sklearn transformers
|
├── documents
|   ├── generated_images          <- Visualizations of model performance, experiments
|   └── readme_images             <- Image assets for README.md
├── results
│   ├── experiments               <- Experiment results
│   ├── logs                      <- TensorBoard logs
│   ├── models                    <- Trained model weights
│   └── predictions               <- Model predictions and explanations
|
├── src
│   ├── custom                    <- Custom TensorFlow components
|   |   └── metrics.py            <- Definition of custom TensorFlow metrics
│   ├── data                      <- Data processing
|   |   ├── client_export.sql     <- SQL query to get raw data from HIFIS database
|   |   └── preprocess.py         <- Preprocessing script
│   ├── interpretability          <- Model interpretability scripts
|   |   └── lime_explain.py       <- Script for generating LIME explanations
│   ├── models                    <- TensorFlow model definitions
|   |   └── models.py             <- Script containing model definition
|   ├── visualization             <- Visualization scripts
|   |   └── visualize.py          <- Script for visualizing model performance metrics
|   ├── horizon_search.py         <- Script for comparing different prediction horizons
|   ├── predict.py                <- Script for prediction on raw data using trained models
|   └── train.py                  <- Script for training model on preprocessed data
|
├── .gitignore                    <- Files to be be ignored by git.
├── config.yml                    <- Values of several constants used throughout project
├── config_private.yml            <- Private information, e.g. database keys (not included in repo)
├── export_clients_to_csv.ps1     <- Powershell script that executes SQL query to get raw data from HIFIS database
├── LICENSE                       <- Project license
├── README.md                     <- Project description
└── requirements.txt              <- Lists all dependencies and their respective versions
```

## Project Config
Many of the components of this project are ready for use on your HIFIS
data. However, this project contains several configurable variables that
are defined in the project config file: [config.yml](config.yml). When
loaded into Python scripts, the contents of this file become a
dictionary through which the developer can easily access its members.

For user convenience, the config file is organized into major steps in
our model development pipeline. Many fields need not be modified by the
typical user, but others may be modified to suit the user's specific
goals. A summary of the major configurable elements in this file is
below.

#### PATHS
- **RAW_DATA**: Path to .csv file generated by running
  _ClientExport.sql_
#### DATA
- **N_WEEKS**: The number of weeks in the future the model will be
  predicting the probability of chronic homelessness (i.e. predictive
  horizon)
- **GROUND_TRUTH_DATE**: Date at which to compute ground truth (i.e.
  state of chronic homelessness) for clients
- **CHRONIC_THRESHOLD**: Number of stays per year for a client to be
  considered chronically homeless
- **FEATURES_TO_DROP_FIRST**: Features you would like to exclude
  entirely from the model. For us, this list evolved through trial and
  error. For example, after running LIME to produce prediction
  explanations, we realized that features in the database that should
  have no impact on the ground truth (e.g. EyeColour) were appearing in
  some explanations; thus, they were added to this list so that these
  problematic correlations and inferences would not be made by the
  model. Incidentally, this iterative feature engineering using LIME
  (explainable AI) to identify bad correlations is the foundation of
  ensuring a machine learning model is free of bias and that its
  predictions are valuable.
- **IDENTIFYING_FEATURES_TO_DROP_LAST**: A list of features that are
  used to preprocess data but are eventually excluded from the
  preprocessed data, as the model cannot consume them. You will not
  likely have to edit this unless you have additional data features
  which are not noted in our config file.
- **TIMED_FEATURES_TO_DROP_LAST**: A list of features containing dates
  that are used in preprocessing but are eventually excluded from the
  preprocessed data. Add any features describing a start or end date to
  this list (e.g. _'LifeEventStartDate'_, _'LifeEventEndDate'_)
- **TIMED_EVENT_FEATURES**: A dictionary where each key is a timestamp
  feature (e.g. _'LifeEventStartDate'_) and every value is a list of
  features that are associated with the timestamp. For example, the
  _'LifeEvent'_ feature is associated with the _'LifeEventStartDate'_
  feature. For paired start and end features, include 1 entry in this
  dictionary for associated features. For example, you need to include
  only one of _'LifeEventStartDate'_ and _'LifeEventEndDate'_ as a key,
  along with _['LifeEvent']_ as the associated value.
- **SPDAT**: Parameters associated with addition of client SPDAT data
  - **INCLUDE_SPDATS**: Boolean variable indicating whether to include
    SPDAT data during preprocessing
  - **SPDAT_CLIENTS_ONLY**: Boolean variable indicating whether to
    include only clients who have a SPDAT in the preprocessed dataset
#### NN
- **MODEL1**: Contains definitions of configurable hyperparameters
  associated with the model architecture. The values currently in this
  section were the optimal values for our dataset informed by a
  hyperparameter search.
#### TRAIN
- **TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT**: Fraction of the data allocated
  to the training, validation and test sets respectively
- **EPOCHS**: Number of epochs to train the model for
- **BATCH_SIZE**: Mini-batch size during training
- **POS_WEIGHT**: Coefficient to multiply the positive class' weight by
  during computation of loss function. Negative class' weight is
  multiplied by (1 - POS_WEIGHT). Increasing this number tends to
  increase recall and decrease precision.
- **IMB_STRATEGY**: Class imbalancing strategy to employ. In our
  dataset, the ratio of positive to negative ground truth was very low,
  prompting the use of these strategies. Set either to _'class_weight'_,
  _'random_oversample'_, _'smote'_, or _'adasyn'_.
- **EXPERIMENT_TYPE**: The type of training experiment you would like to
  perform if executing [_train.py_](src/train.py). Choices are
  _'single_train'_, _'multi_train'_, or _'hparam_search'_.
- **METRIC_MONITOR**: The metric to monitor when training multiple
  models serially (i.e. the _'multi_train'_ experiment in
  [_train.py_](src/train.py))
- **NUM_RUNS**: The number of times to train a model in the
  _'multi_train'_ experiment
- **THRESHOLDS**: A single float or list of floats in range [0, 1]
  defining the classification threshold. Affects precision and recall
  metrics.
- **HP**: Parameters associated with random hyperparameter search
  - **METRICS**: List of metrics on validation set to monitor in
    hyperparameter search. Can be any combination of _{'accuracy',
    'loss', 'recall', 'precision', 'auc'}_
  - **COMBINATIONS**: Number of random combinations of hyperparameters
    to try in hyperparameter search
  - **REPEATS**: Number of times to repeat training per combination of
    hyperparameters
  - **RANGES**: Ranges defining possible values that hyperparameters may
    take. Be sure to check [_train.py_](src/train.py) to ensure that
    your ranges are defined correctly as real or discrete intervals (see
    [Random Hyperparameter Search](#random-hyperparameter-search) for an
    example).
#### LIME
- **KERNEL_WIDTH**: Affects size of neighbourhood around which LIME
  samples for a particular example. In our experience, setting this
  within the continuous range of _[1.0, 2.0]_ is large enough to produce
  stable explanations, but small enough to avoid producing explanations
  that approach a global surrogate model.
- **FEATURE_SELECTION**: The strategy to select features for LIME
  explanations. Read the LIME creators'
  [documentation](https://lime-ml.readthedocs.io/en/latest/lime.html)
  for more information. **NUM_FEATURES**: The number of features to
  include in a LIME explanation **NUM_SAMPLES**: The number of samples
  used to fit a linear model when explaining a prediction using LIME
#### HORIZON_SEARCH
- **N_MIN**: Smallest prediction horizon to use in the prediction
  horizon search experiment (in weeks)
- **N_MAX**: Largest prediction horizon to use in the prediction horizon
  search experiment (in weeks)
- **N_INTERVAL**: Size of increment to increase the prediction horizon
  by when iterating through possible prediction horizons (in weeks)
- **RUNS_PER_N**: The number of times to train the model per value of
  the prediction horizon in the experiment
#### PREDICTION
- **THRESHOLD**: Classification threshold for prediction
- **CLASS_NAMES**: Identifiers for the classes predicted by the neural
  network as included in the prediction spreadsheet.

## Contact

**Matt Ross**  
Manager, Artificial Intelligence  
Information Technology Services, City Manager’s Office  
City of London  
201 Queens Ave. Suite 300, London, ON. N6A 1J1  
P: 519.661.CITY (2489) x 5451 | C: 226.448.9113



