# HIFIS-v2 Model

The purpose of this project is to investigate the efficacy of a machine
learning solution to assist in identifying individuals at risk of
chronic homelessness. A model prototype was built for the City of
London, Ontario, Canada. This repository contains the code used to train
a neural network model to classify clients in the city's HIFIS database
as either at risk or not at risk of chronic homelessness within a
specified predictive horizon. In an effort to comply with the Canadian
federal government's
[Directive on Automated Decision-Making](https://www.tbs-sct.gc.ca/pol/doc-eng.aspx?id=32592),
this repository applies interpretability methods to explain the model's
predictions. to This repository is intended to serve as a template for
other municipalities who wish to explore the application of this model
in their own locales.

## Getting Started
1. Clone this repository (for help see this
   [tutorial](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)).
2. Open [_export_clients_to_csv.ps1_](export_clients_to_csv.ps1) for
   editing. Replace "_[Instance Name goes here]_" with your HIFIS
   database instance name. Execute
   [_export_clients_to_csv.ps1_](export_clients_to_csv.ps1). A file
   named _"HIFIS_Clients.csv"_ should now be within the _data/raw/_
   folder.
3. Check that your features in _HIFIS_Clients.csv_ match in
   [config.yml](config.yml). If necessary, update feature
   classifications in this file (for help see the [section on project
   config](#project-config).
4. Execute [_preprocess.py_](src/data/preprocess.py) to transform the
   data into the format required by the machine learning model.
   Preprocessed data will be saved within _data/preprocessed/_.
5. Execute [_train.py_](src/train.py) to train the neural network model
   on your preprocessed data. The trained model weights will saved
   within _results/models/_, and the TensorBoard log files will be saved
   within _results/logs/_.
6. Execute [_lime_explain.py_](src/interpretability/lime_explain.py) to
   generate interpretable explanations for the model's predictions on
   the test set. A spreadsheet of predictions and explanations will be
   saved within _results/experiments/_.

## Project Structure
The project looks similar to the directory structure below. Disregard
any _.gitkeep_ files, as they exist solely to demonstrate preferred
project organization.

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
│   └──  predictions              <- Model predictions and explanations
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
└── README.md                     <- Project description
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
goals. A summary of the major configurable elements in this file
follows.

#### PATHS
- **RAW_DATA**: Path to .csv file generated by running
  _ClientExport.sql_
#### DATA
- **N_WEEKS**: Predictive horizon in weeks
- **GROUND_TRUTH_DATE**: Date at which to compute ground truth (i.e.
  state of chronic homelessness) for clients
- **CHRONIC_THRESHOLD**: Number of stays per year for a client to be
  considered chronically homeless
- **FEATURES_TO_DROP_FIRST**: Features you would like to exclude
  entirely from the model. For us, this list evolved with trial and
  error. For example, after running LIME to produce prediction
  explanations, we realized that features in the database that should
  have no impact on the ground truth (e.g. EyeColour) were appearing in
  some explanations; thus, they were added to this list.
- **IDENTIFYING_FEATURES_TO_DROP_LAST**: A list of features that are
  used to preprocess data but are eventually excluded from the
  preprocessed data. You will not likely have to edit this.
- **TIMED_FEATURES_TO_DROP_LAST**: A list of features containing dates
  that are used in preprocessing but are eventually excluded from the
  preprocessed data. Add any features describing a start or end date to
  this list (e.g. _'LifeEventStartDate'_, _'LifeEventEndDate'_
- **TIMED_EVENT_FEATURES**: A dictionary where each key is a timestamp
  feature (e.g. _'LifeEventStartDate'_) and every value is a list of
  features that are associated with the timestamp. For example, the
  _'LifeEvent'_ feature is associated with the _'LifeEventStartDate'_
  feature. For paired start and end features, include 1 entry in this
  dictionary for associated features. For example, you need to include
  only one of _'LifeEventStartDate'_ and _'LifeEventEndDate'_ as a key,
  along with _['LifeEvent']_ as the associated value.
#### NN
- **MODEL1**: Contains definitions of configurable hyperparameters
  associated with the model architecture. The values currently in this
  section were informed by a hyperparameter search.
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
  propmting the use of these strategies. Set either to _'class_weight'_,
  _'random_oversample'_, _'smote'_, or _'adasyn'_.
- **EXPERIMENT_TYPE**: The type of training experiment you would like to
  perform if executing [_train.py_](src/train.py). Choices are
  _'single_train'_, _'multi_train'_, or _'hparam_search'_.
- **METRIC_MONITOR**: The metric to monitor when training multiple
  models serially (i.e. the _'multi_train'_ experiment in
  [_train.py_](src/train.py)
- **NUM_RUNS**: The number of times to train a model in the
  _'multi_train'_ experiment
- **THRESHOLDS**: A single float or list of floats in range [0, 1]
  defining the classification threshold. Affects precision and recall
  metrics.
- **HP**: Parameters associated with random hyperparameter search
  - **METRICS**: List of metrics on validation set to monitor in
    hyperparameter search
  - **COMBINATIONS**: Number of random combinations of hyperparameters
    to try in hyperparameter search
  - **REPEATS**: Number of times to repeat training per combination of
    hyperparameters
  - **RANGES**: Ranges defining possible values that
    hyperparameters may take. Be sure to check
    [_train.py_](src/train.py) to ensure that your ranges are defined
    correctly as real or discrete intervals.
#### LIME
- **KERNEL_WIDTH**: Affects size of neighbourhood around which LIME
  samples for a particular example
- **FEATURE_SELECTION**: The strategy to select features for LIME
  explanations. Read the LIME creators'
  [documentation](https://lime-ml.readthedocs.io/en/latest/lime.html)
  for more information.
  **NUM_FEATURES**: The number of features to include in a LIME
  explanation **NUM_SAMPLES**: The number of samples used to fit a
  linear model when explaining a prediction using LIME
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

## Use Cases

### Train a model and visualize results.
1. Once you have _HIFIS_Clients.csv_ sitting in the raw data folder
   (_data/raw/), execute [_preprocess.py_](src/data/preprocess.py). See
   [Getting Started](#getting-started) for help obtaining
   _HIFIS_Clients.csv_.
2. Ensure data has been preprocessed properly. That is, verify that
   _data/processed/_ contains both _HIFIS_Processed.csv_ and
   _HIFIS_Processed_OHE.csv_. The the latter is the same as the former
   with the exception being that its single-valued categorical features
   have been one-hot encoded.
3. In [config.yml](config.yml), set _EXPERIMENT_TYPE_ within _TRAIN_ to
   _'single_train'_.
4. Execute [train.py](src/train.py). The trained model's weights will be
   located in _results/models/_, and its filename will resemble the
   following structure: modelyyyymmdd-hhmmss.h5, where yyyymmdd-hhmmss
   is the current time. The model's logs will be located in
   _results/logs/_, and its directory name will be the current time in
   the same format. These logs contain information about the experiment,
   such as metrics throughout the training process on the training and
   validation sets, and performance on the test set. The logs can be
   visualized by running
   [TensorBoard](https://www.tensorflow.org/tensorboard/get_started)
   locally. See below for an example of a plot from a TensorBoard log
   file.

![alt text](documents/readme_images/tensorboard_example.PNG "Loss vs
Epoch")

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
business decision as to which value yields acceptable results. See below
for instructions on how to run a Prediction Horizon Search Experiment.
1. In the _HORIZON_SEARCH_ section of [config.yml](config.yml), set
   _N_MIN_, _N_MAX_, _N_INTERVAL_ and _RUNS_PER_N_ according to your
   organization's needs (see [Project Config](#project-config) for help).
2. Run _src/horizon_search.py_. This may take several minutes to hours,
   depending on your hardware and settings from the previous step.
3. A .csv representation of experiment results will be available within
   _results/experiments/, called _horizon_searchyyyymmdd-hhmmss.csv_,
   where yyyymmdd-hhmmss is the current time. A graphical representation
   of the results will be available within
   _documents/generated_images/_, called
   _horizon_experiment_yyyymmdd-hhmmss.png_. See below for an example of
   this visualization.

![alt text](documents/readme_images/horizon_experiment_example.png
"Prediction Horizon Search Experiment")

### LIME Experiment
Since the predictions made by this model are to be used by a government
institution to benefit vulnerable members of society, it is imperative
that the model's predictions may be explained somehow. Since this model
is a neural network, it is difficult to decipher which rules or
heuristics it is employing to make its predictions. Interpretability in
machine learning is a growing concern, especially with applications in
the healthcare and social services domains. We used [Local Interpretable
Model-Agnostic Explanations](https://arxiv.org/pdf/1602.04938.pdf) (i.e.
LIME) to explain the predictions of the neural network classifier that
we trained. We used the implementation available in the authors' [GitHub
repository](https://github.com/marcotcr/lime). LIME perturbs the
features in an example and trains a linear model to approximate the
neural network at the local region in the feature space surrounding the
example. It then uses the linear model to determine which features were
most contributory to the model's prediction for that example. By
applying LIME to our trained model, we can conduct informed feature
engineering based on any obviously inconsequential features we see (e.g.
EyeColour) or insights from domain experts. We can also tell if the
model is learning some sort of unwanted bias. See the steps below to
apply LIME to explain the model's predictions on examples in the test
set.
1. Having previously run _[train.py](src/train.py)_, ensure that
   _data/processed/_ contains both _Train_Set.csv_ and _Test_Set.csv_.
2. In the _main_ function of
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
      will run LIME on an the example in the test set whose ClientID is
      that which you passed to the function. A graphic will be generated
      that will depict the top explainable features that the model used
      to make its prediction. See below for an example of this graphic.
3. Interpret the output of the LIME explainer. LIME partitions features
   into classes or ranges and reports the features most contributory to
   a prediction. A feature explanation is considered to be a value (or
   range of values) of a feature and its associated weight in the
   prediction. In the example below, the fact that TotalStays was
   greater than 4 but less than or equal to 23 contributed negatively
   with a magnitude of about 0.22 to a positive prediction (meaning it
   contributed at a magnitude of 0.22 toward a negative prediction). As
   another example, the rule "ReasonForService_Streets=1" indicates that
   at some point the client has a record that cites their reason for
   service as "Streets" (_=1_ indicates that a boolean feature is
   present, and _=0_ indicates that a boolean feature is not present)
   and that this explanation contributed with a weight of about 0.02
   toward a positive prediction. As one last example, consider that this
   client's AboriginalIndicator value is "Yes - Tribe Not Known", which
   contributed with a weight of about 0.04 towards a negative
   prediction.

![alt text](documents/readme_images/LIME_example.PNG "A sample LIME
explanation")



