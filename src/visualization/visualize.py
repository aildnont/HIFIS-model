from sklearn.metrics import confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import stats
import datetime

# Set some matplotlib parameters
mpl.rcParams['figure.figsize'] = (12, 10)


def plot_metrics(history, metrics, file_path=None):
    '''
    Plot metrics for the training and validation sets over the training history.
    :param history: Model history; returned from model.fit()
    :param metrics: List of metrics to plot
    '''
    plt.clf()
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
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
    if file_path is not None:
        plt.savefig(file_path + 'metrics_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    return

def plot_roc(name, labels, predictions, file_path=None):
    '''
    Plots the ROC curve for predictions on a dataset
    :param name: Name of dataset on the plot
    :param labels: Ground truth labels
    :param predictions: Model predictions corresponding to the labels
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
    if file_path is not None:
        plt.savefig(file_path + 'ROC_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    return

def plot_confusion_matrix(labels, predictions, p=0.5, file_path=None):
    '''
    Plot a confusion matrix for the ground truth labels and corresponding model predictions.
    :param labels: Ground truth labels
    :param predictions: Model predictions
    :param p: Classification threshold
    '''
    plt.clf()
    fig, ax = plt.subplots()
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
    if file_path is not None:
        plt.savefig(file_path + 'CM_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')

    # Print these statistics
    print('True (-)ves: ', cm[0][0], '\nFalse (+)ves: ', cm[0][1], '\nFalse (-)ves: ', cm[1][0], '\nTrue (+)ves: ',
          cm[1][1])
    return

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