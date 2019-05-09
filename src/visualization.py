from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from functools import wraps
from utils import flatten
import seaborn as sns
import numpy as np

def _flattens_y(func):
    @wraps(func)
    def wrapper(y_true, y_pred, *args, **kwargs):
        y_true_flat = flatten(y_true)
        y_pred_flat = flatten(y_pred)
        return func(y_true_flat, y_pred_flat, *args, **kwargs)
    return wrapper

@_flattens_y
def plot_confusion_matrix(y_true, y_pred, labels):

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(16, 12))

    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, fmt="d", cmap="YlGnBu", ax=ax)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return ax
