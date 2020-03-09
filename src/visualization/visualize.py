from sklearn.metrics import confusion_matrix, roc_curve
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import stats
import pandas as pd
import datetime
import io
import os
import yaml
from math import floor, ceil

# Set some matplotlib parameters
mpl.rcParams['figure.figsize'] = (12, 10)

def plot_to_tensor():
    '''
    Converts a matplotlib figure to an image tensor
    :param figure: A matplotlib figure
    :return: Tensorflow tensor representing the matplotlib image
    '''
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)     # Convert .png buffer to tensorflow image
    image = tf.expand_dims(image, 0)     # Add the batch dimension
    return image

def plot_metrics(history, metrics, dir_path=None):
    '''
    Plot metrics for the training and validation sets over the training history.
    :param history: Model history; returned from model.fit()
    :param metrics: List of metrics to plot
    :param dir_path: Directory in which to save image
    '''
    plt.clf()
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,3,n+1)
        plt.plot(history.epoch,  history.history[metric], label='Train')    # Plot metric on training data
        plt.plot(history.epoch, history.history['val_'+metric], linestyle="--", label='Val')    # Plot metric on validation data
        plt.xlabel('Epoch')
        plt.ylabel(name)

        # Set plot limits depending on the metric
        if metric == 'loss':
          plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
          plt.ylim([0.8,1])
        else:
          plt.ylim([0,1])
        plt.legend()
    if dir_path is not None:
        plt.savefig(dir_path + 'metrics_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    return

def plot_roc(name, labels, predictions, dir_path=None):
    '''
    Plots the ROC curve for predictions on a dataset
    :param name: Name of dataset on the plot
    :param labels: Ground truth labels
    :param predictions: Model predictions corresponding to the labels
    :param dir_path: Directory in which to save image
    '''
    plt.clf()
    fp, tp, _ = roc_curve(labels, predictions)  # Get values for true positive and true negative
    plt.plot(100*fp, 100*tp, label=name, linewidth=2)   # Plot the ROC curve
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,20])
    plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    if dir_path is not None:
        plt.savefig(dir_path + 'ROC_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    return plot_to_tensor()

def plot_confusion_matrix(labels, predictions, p=0.5, dir_path=None):
    '''
    Plot a confusion matrix for the ground truth labels and corresponding model predictions.
    :param labels: Ground truth labels
    :param predictions: Model predictions
    :param p: Classification threshold
    :param dir_path: Directory in which to save image
    '''
    plt.clf()
    ax = plt.subplot()
    cm = confusion_matrix(labels, predictions > p)  # Calculate confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # Plot confusion matrix
    ax.figure.colorbar(im, ax=ax)
    ax.set(yticks=[-0.5, 1.5], xticks=[0, 1], yticklabels=['0', '1'], xticklabels=['0', '1'])
    ax.yaxis.set_major_locator(mpl.ticker.IndexLocator(base=1, offset=0.5))

    # Print number of TPs, FPs, TNs, FNs on each quadrant in the plot
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    # Set plot's title and axis names
    plt.title('Confusion matrix p={:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    # Save the image
    if dir_path is not None:
        plt.savefig(dir_path + 'CM_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')

    # Print these statistics
    print('True (-)ves: ', cm[0][0], '\nFalse (+)ves: ', cm[0][1], '\nFalse (-)ves: ', cm[1][0], '\nTrue (+)ves: ',
          cm[1][1])
    return plot_to_tensor()

def plot_horizon_search(results_df, file_path):
    '''
    Plots the results of a prediction horizon search experiment.
    Each metric applied to the test set is plotted over a varying prediction horizon
    :param results_df: A Pandas dataframe containing the results of a search experiment
    :param file_path: The path to save an image of the plot
    '''
    num_metrics = len(results_df.columns) - 1   # Number of metrics to plot
    average_df = results_df.groupby('n', as_index=False).mean() # Get the mean values for each metric by value of n
    fig, axes = plt.subplots(nrows=int(num_metrics ** 0.5), ncols=int(num_metrics ** 0.5) + 1, constrained_layout=True)
    axes = axes.flat
    for i in range(num_metrics):
        metric_name = results_df.columns[i + 1].replace("_"," ").capitalize()

        # Plot the results for each training run, along with a trendline.
        axes[i].scatter(results_df['n'],  results_df[results_df.columns[i + 1]], label='Run Results', marker='o', s=10, color='blue')

        # Compute and plot a least squares regression for this metric
        slope, intercept, r_value, p_value, std_err = stats.linregress(results_df['n'], results_df[results_df.columns[i + 1]])
        trend = np.poly1d([slope, intercept])
        axes[i].plot(results_df['n'], trend(results_df['n']), label='Linear Fit', linestyle='-', color='blue')
        axes[i].text(0.05,0.95,"R^2 = {:.2f}".format(r_value ** 2), transform=axes[i].transAxes)    # Print R^2 value

        # Plot the average of training runs for each n
        axes[i].plot(average_df['n'],  average_df[average_df.columns[i + 1]], label='Average by n', linestyle='--', color='green')

        # Set plot labels and legend
        axes[i].set_xlabel('Prediction horizon [weeks]')
        axes[i].set_ylabel(metric_name)
        axes[i].legend()
    fig.suptitle('Test Set Metrics for Training Runs with Varying Prediction Horizons')
    for axis in axes[num_metrics:]:
        fig.delaxes(axis)   # Delete blank axis

    if file_path is not None:
        plt.savefig(file_path + 'horizon_experiment_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    return

def shorten_explanations(exps, max_exp_len=87):
    '''
    Given a list of explanation labels, shorten them by replacing the middle portion of longer ones with an ellipsis
    :param exps: List of explanation labels
    :param max_exp_len: Maximum desired length of an explanation label
    :return: List of shortened explanation labels
    '''
    half_exp_len = (max_exp_len - 7) // 2   # Calculate half of desired max length. Note: an ellipsis is 7 chars long

    # If an explanation label is too long, take out the middle characters and replace with an ellipsis
    for i in range(len(exps)):
        if len(exps[i]) >= max_exp_len:
            exps[i] = exps[i][0:half_exp_len] + ' . . . ' + exps[i][-half_exp_len:]
    return exps

def visualize_explanation(explanation, client_id, client_gt):
    '''
    Visualize top LIME contributing features for an example.
    :param explanation: Local explanation of example
    :param client_id: ClientID of example
    :param ground_truth: GroundTruth of example
    '''
    fig = explanation.as_pyplot_figure()
    probs = explanation.predict_proba
    fig.text(0.02, 0.98, "Prediction probabilities: ['0': {:.2f}, '1': {:.2f}]".format(probs[0], probs[1]))
    fig.text(0.02, 0.96, "Client ID: " + str(client_id))
    fig.text(0.02, 0.94, "Ground Truth: " + str(client_gt))

    # Shorten explanation rules that are too long to be graph labels
    ax = plt.gca()
    exps = shorten_explanations([t._text for t in ax.get_yticklabels()], max_exp_len=87)
    ax.set_yticklabels(exps)
    plt.tight_layout()

def explanations_to_hbar_plot(exp_weights, title='', subtitle=''):
    '''
    Plot a series of explanations and weights, sorted by weights, on a horizontal bar graph
    :param exp_weights: list of of (explanation, weight) tuples
    :param title: Title of horizonal bar graph
    '''

    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))    # Load project config
    half_max_exps = cfg['LIME']['MAX_DISPLAYED_RULES'] // 2

    # Separate and sort positively and negatively weighted explanations
    pos_exp_weights = [e_w for e_w in exp_weights if e_w[1] >= 0]
    neg_exp_weights = [e_w for e_w in exp_weights if e_w[1] < 0]
    pos_exp_weights = sorted(pos_exp_weights, key=lambda e: e[1], reverse=True)
    neg_exp_weights = sorted(neg_exp_weights, key=lambda e: e[1], reverse=False)

    # Limit number of explanations shown based on config file and preserve ratio of positive weights to negative weights
    ratio = len(pos_exp_weights) / len(neg_exp_weights)
    if ratio >= 1:
        if len(pos_exp_weights) > half_max_exps:
            pos_exp_weights = pos_exp_weights[0:half_max_exps]
        max_neg_exps = floor(len(pos_exp_weights) / ratio)
        if len(neg_exp_weights) > max_neg_exps:
            neg_exp_weights = neg_exp_weights[0:max_neg_exps]
    else:
        if len(neg_exp_weights) > half_max_exps:
            neg_exp_weights = neg_exp_weights[0:half_max_exps]
        max_pos_exps = floor(len(neg_exp_weights) * ratio)
        if len(pos_exp_weights) > max_pos_exps:
            pos_exp_weights = pos_exp_weights[0:max_pos_exps]

    # Assemble final list of explanations and weights
    exp_weights = pos_exp_weights + neg_exp_weights

    # Sort by value of weights
    exp_weights = sorted(exp_weights, key=lambda e: e[1])

    # Get corresponding lists for explanations and weights
    exps = [e for e, w in exp_weights]
    weights = [w for e, w in exp_weights]

    # Shorten explanation rules that are too long to be graph labels
    exps = shorten_explanations(exps, max_exp_len=87)

    fig, axes = plt.subplots(constrained_layout=True)
    ax = plt.subplot()
    colours = ['green' if x > 0 else 'red' for x in weights]    # Colours for positive and negative weights
    positions = np.arange(len(exps))    # Positions for bars on y axis
    ax.barh(positions, weights, align='center', color=colours)  # Plot a horizontal bar graph of the average weights

    # Print the average weight in the center of its corresponding bar
    max_weight = abs(max(weights, key=abs))
    for bar, weight in zip(ax.patches, weights):
        if weight >= 0:
            ax.text(bar.get_x() - max_weight * 0.09, bar.get_y() + bar.get_height() / 2, '{:.3f}'.format(weight),
                    color='green', ha='center', va='center', fontweight='semibold', transform=ax.transData)
        else:
            ax.text(bar.get_x() + max_weight * 0.09, bar.get_y() + bar.get_height() / 2, '{:.3f}'.format(weight),
                    color='red', ha='center', va='center', fontweight='semibold', transform=ax.transData)

    # Set ticks for x and y axes. For x axis, set major and minor ticks at intervals of 0.05 and 0.01 respectively.
    ax.set_yticks(positions)
    ax.set_yticklabels(exps, fontsize=8, fontstretch='extra-condensed')
    ax.set_xticks(np.arange(floor(min(weights)/0.05)*0.05, ceil(max(weights)/0.05)*0.05 + 0.05, 0.05), minor=False)
    ax.set_xticks(np.arange(floor(min(weights)/0.05)*0.05, ceil(max(weights)/0.05)*0.05 + 0.05, 0.01), minor=True)
    plt.xticks(rotation=45, ha='right', va='top')

    # Display a grid behind the bars.
    ax.grid(True, which='major')
    ax.grid(True, which='minor', axis='x', linewidth=1, linestyle=':')
    ax.set_axisbelow(True)

    # Set plot axis labels, title, and subtitle.
    ax.set_xlabel("Contribution to Probability of Chronic Homelessness", labelpad=10, size=15)
    ax.set_ylabel("Feature Explanations", labelpad=10, size=15)
    fig.suptitle(title, size=20)                                 # Title
    fig.text(0.5, 0.9, subtitle, size=15, ha='center')          # Subtitle
    fig.set_constrained_layout_pads(w_pad=0.25, h_pad=0.25)
    return


def visualize_avg_explanations(results_df, sample_fraction, file_path=None):
    '''
    Builds a graph for visualizing the average weights of LIME explanations over all provided explanations
    :param results_df: Output dataframe of running a LIME experiment to explain multiple predictions
    :param sample_fraction: Fraction of test set that explanations were produced for
    :param file_path: The path to the directory at which to save the resulting image
    '''

    # Concatenate all feature explanations and their corresponding weights for each example
    exp_cols = [col for col in results_df.columns if ('Exp' in col)]
    weight_cols = [col for col in results_df.columns if ('Weight' in col)]
    exps = np.concatenate([results_df[[exp_cols[i], weight_cols[i]]].values for i in range(len(exp_cols))], axis=0)
    exps_df = pd.DataFrame(exps, columns=['Exp', 'Weight']).astype({'Weight' : 'float64'})

    # Compute the average weight for each distinct feature explanation (e.g. TotalScore > 0)
    avg_exps = exps_df.groupby('Exp', as_index=False).agg({'Weight' : np.mean}).sort_values(by='Weight').values
    exps = [x[0] for x in avg_exps]
    weights = [x[1] for x in avg_exps]
    exp_data = list(zip(exps, weights))

    # Plot as horizontal bar graph
    title = 'Average Weights for LIME Explanations on Test Set'
    subtitle = '% of test set sampled = ' + '{:.2f}'.format(sample_fraction * 100)
    explanations_to_hbar_plot(exp_data, title=title, subtitle=subtitle)

    # Save the image
    if file_path is not None:
        plt.savefig(file_path + 'LIME_Explanations_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    return

def visualize_submodular_pick(W_avg, sample_fraction, file_path=None):
    '''
    Builds a graph for visualizing the average weights of a set of LIME explanations resulting from a submodular pick
    :param W_avg: A dataframe containing average LIME explanations from a submodular pick
    :param sample_fraction: Fraction of training set examples explained for submodular pick
    :param file_path: The path to the directory at which to save the resulting image
    '''

    # Sort by weight
    W_avg = W_avg.sort_values(ascending=True)
    exps = W_avg.index.tolist()
    weights = W_avg.tolist()
    exp_data = list(zip(exps, weights))

    # Plot as horizontal bar graph
    title = 'Average Weights for Explanations from Submodular Pick'
    subtitle = '% of training set sampled = ' + '{:.2f}'.format(sample_fraction * 100)
    explanations_to_hbar_plot(exp_data, title=title, subtitle=subtitle)

    # Save the image
    if file_path is not None:
        plt.savefig(file_path + 'LIME_Submodular_Pick_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    return
