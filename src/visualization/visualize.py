from sklearn.metrics import confusion_matrix, roc_curve
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvas
import numpy as np
from scipy import stats
import pandas as pd
import datetime
import io
import os
import yaml
from math import floor, ceil, sqrt

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
    return plt

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
    return plt

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


def visualize_explanation(explanation, client_id, client_gt, date=None, file_path=None):
    '''
    Visualize top LIME contributing features for an example.
    :param explanation: Local explanation of example
    :param client_id: ClientID of example
    :param ground_truth: GroundTruth of example
    :param date: Date of example
    :param file_path: The path to the directory at which to save the resulting image
    '''

    # Create horizontal bar graph for the explanation
    fig = explanation.as_pyplot_figure()
    probs = explanation.predict_proba
    fig.text(0.02, 0.98, "Prediction probabilities: ['0': {:.2f}, '1': {:.2f}]".format(probs[0], probs[1]), fontsize='x-small')
    fig.text(0.02, 0.96, "Ground Truth: " + str(client_gt), fontsize='x-small')
    fig.text(0.02, 0.94, "Client ID: " + str(client_id), fontsize='x-small')
    if date is not None:
        fig.text(0.02, 0.92, "Date: " + date, fontsize='x-small')
    plt.tight_layout()

    # Save the image
    if file_path is not None:
        plt.savefig(file_path + 'Client_' + str(client_id) + '_exp_' +
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    return

def visualize_multiple_explanations(explanations, title):
    '''
    Create a single figure containing bar charts of a list of explanations.
    :param explanations: List of Explanation objects
    :param title: Plot title
    :param file_path: The path (including file name) where to save the resulting image
    '''

    # Create a figure for each explanation and get its image buffer
    n_exps = len(explanations)
    fig_imgs = []
    for i in range(n_exps):
        exp_fig_canvas = FigureCanvas(explanations[i].as_pyplot_figure())
        exp_fig_canvas.draw()   # Force a draw to get access to pixel buffer
        fig_imgs.append(np.array(exp_fig_canvas.renderer.buffer_rgba()))

    # Plot all explanation graphs on the same figure
    plt.clf()
    n_cols = ceil(sqrt(n_exps))
    n_rows = floor(sqrt(n_exps))
    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows)
    fig.set_size_inches(60, 35)
    for i in range(n_rows * n_cols):
        if i < n_exps:
            axes.ravel()[i].imshow(fig_imgs[i])     # Display explanation image on axes
        axes.ravel()[i].set_axis_off()              # Don't show x-axis and y-axis
    fig.tight_layout(pad=0.01, h_pad=0.01, w_pad=0.01)
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
    fig.suptitle(title, size=60)
    return fig


def explanations_to_hbar_plot(exp_weights, title='', subtitle=''):
    '''
    Plot a series of explanations and weights, sorted by weights, on a horizontal bar graph
    :param exp_weights: list of of (explanation, weight) tuples
    :param title: Title of horizonal bar graph
    '''

    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))    # Load project config
    half_max_exps = cfg['LIME'][cfg['TRAIN']['MODEL_DEF'].upper()]['MAX_DISPLAYED_RULES'] // 2

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
    for i in range(len(exps)):
        if len(exps[i]) >= 87:
            exps[i] = exps[i][0:40] + ' . . . ' + exps[i][-40:]

    fig, axes = plt.subplots(constrained_layout=True)
    ax = plt.subplot()
    colours = ['green' if x > 0 else 'red' for x in weights]    # Colours for positive and negative weights
    positions = np.arange(len(exps))    # Positions for bars on y axis
    ax.barh(positions, weights, align='center', color=colours)  # Plot a horizontal bar graph of the average weights

    # Print the average weight in the center of its corresponding bar
    max_weight = abs(max(weights, key=abs))
    for bar, weight in zip(ax.patches, weights):
        if weight >= 0:
            ax.text(bar.get_x() - max_weight * 0.1, bar.get_y() + bar.get_height() / 2, '{:.3f}'.format(weight),
                    color='green', ha='center', va='center', fontweight='semibold', transform=ax.transData)
        else:
            ax.text(bar.get_x() + max_weight * 0.1, bar.get_y() + bar.get_height() / 2, '{:.3f}'.format(weight),
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
    fig.text(0.5, 0.92, subtitle, size=15, ha='center')          # Subtitle
    fig.set_constrained_layout_pads(w_pad=0.25, h_pad=0.25)
    return


def visualize_avg_explanations(results_df, sample_fraction, file_path=None):
    '''
    Builds a graph for visualizing the average weights of LIME explanations over all provided explanations
    :param results_df: Output dataframe of running a LIME experiment to explain multiple predictions
    :param sample_fraction: Fraction of test set that explanations were produced for
    :param file_path: The path at which to save the resulting image
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
    subtitle = '% of training and validation examples sampled = ' + '{:.2f}'.format(sample_fraction * 100)
    explanations_to_hbar_plot(exp_data, title=title, subtitle=subtitle)

    # Save the image
    if file_path is not None:
        plt.savefig(file_path)
    return


def visualize_cluster_explanations(explanations, predictions, cluster_freqs, title, file_path=None):
    '''
    Create a single figure containing bar charts of a list of explanations of clusters.
    :param explanations: List of Explanation objects
    :param predictions: a (# clusters x 2) numpy array detailing predicted classes and probability of chronic
                        homelessness
    :param cluster_freqs: List of fractions of clients belonging to each cluster
    :param title: Plot title
    :param file_path: The path (including file name) where to save the resulting image
    '''
    fig = visualize_multiple_explanations(explanations, title)  # Generate figure to show all centroid explanations

    # Set title for each explanation graph according to cluster # and % of clients it contains
    for i in range(len(explanations)):
        fig.axes[i].text(0.5, 0.92, 'Cluster ' + str(i + 1) + ' (' + '{:.2f}'.format(cluster_freqs[i + 1] * 100) +
                                      '% of Clients)', fontsize=35, transform=fig.axes[i].transAxes, horizontalalignment='center')
        fig.axes[i].text(0.02, 0.96, "Probability of chronic homelessness: {:.2f}%".format(predictions[i][1]),
                         fontsize=15, transform=fig.axes[i].transAxes)
        fig.axes[i].text(0.02, 0.98, "Prediction: " + str(predictions[i][0]) + " of chronic homelessness", fontsize=15,
                         transform=fig.axes[i].transAxes)

    # Save the image
    if file_path is not None:
        plt.savefig(file_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    return


def visualize_silhouette_plot(k_range, silhouette_scores, optimal_k, file_path=None):
    '''
    Plot average silhouette score for all samples at different values of k. Use this to determine optimal number of
    clusters (k). The optimal k is the one that maximizes the average Silhouette Score over the range of k provided.
    :param k_range: Range of k explored
    :param silhouette_scores: Average Silhouette Score corresponding to values in k range
    :param optimal_k: The value of k that has the highest average Silhouette Score
    '''

    # Plot the average Silhouette Score vs. k
    axes = plt.subplot()
    axes.plot(k_range, silhouette_scores)

    # Set plot axis labels, title, and subtitle.
    axes.set_xlabel("k (# of clusters)", labelpad=10, size=15)
    axes.set_ylabel("Average Silhouette Score", labelpad=10, size=15)
    axes.set_xticks(k_range, minor=False)
    axes.axvline(x=optimal_k, linestyle='--')
    axes.set_title("Silhouette Plot", fontsize=25)
    axes.text(0.5, 0.92, "Average Silhouette Score over a range of k-values", size=15, ha='center')

    # Save the image
    if file_path is not None:
        plt.savefig(file_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    return