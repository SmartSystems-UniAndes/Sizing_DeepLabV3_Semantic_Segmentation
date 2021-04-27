import os
import matplotlib.pyplot as plt
import numpy as np


def plot_hist(hists, labels):
    plt.figure()
    for hist in hists:
        plt.plot(range(len(hist)), hist)
    plt.legend(labels)
    plt.grid()
    plt.xlabel("Epoch")
    plt.show()

    return None


def generate_matrix(y_true, y_hat, n_class):
    mask = (y_true >= 0) & (y_true < n_class)
    hist = np.bincount(n_class*y_true[mask].astype(int) + y_hat[mask], minlength=n_class ** 2).reshape(n_class, n_class)

    return hist


def get_metrics(y_true, y_hat, n_class):
    hist = np.zeros((n_class, n_class))

    for y_true_, y_hat_ in zip(y_true, y_hat):
        hist += generate_matrix(y_true_.flatten(), y_hat_.flatten(), n_class)

    acc = np.diag(hist).sum() / hist.sum()

    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iou = np.nanmean(iou)

    freq = hist.sum(axis=1) / hist.sum()
    freq_weighted_avg_acc = (freq[freq > 0] * iou[freq > 0]).sum()

    return acc, acc_cls, mean_iou, freq_weighted_avg_acc


def save_predictions(loader, x, y_hat, tag, path):
    _, mask = loader.dataset.un_transforms(x, y_hat)
    file = os.path.join(path, tag)
    mask.save(file)

    return True
