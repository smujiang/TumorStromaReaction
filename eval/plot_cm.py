import numpy as np


def plot_confusion_matrix(cm, save_to,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(7.2, 6), dpi=300)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18)
    # plt.figtext(.5, 0.9, title, fontsize=18, ha='center')
    # plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, fontsize=16)
        plt.yticks(tick_marks, target_names, fontsize=16)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=18)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)

    # plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), fontsize=18)
    # plt.show()
    plt.savefig(save_to)

fib = [[0.92, 0.07, 0.00],
       [0.04, 0.92, 0.04],
       [0.00, 0.09, 0.91]]

cell = [[0.90, 0.05, 0.05],
        [0.07, 0.80, 0.14],
        [0.02, 0.03, 0.95]]

ori = [[0.74, 0.15, 0.11],
       [0.12, 0.79, 0.10],
       [0.02, 0.03, 0.95]]

#
# fib = [[0.76, 0.21, 0.02],
#        [0.07, 0.87, 0.06],
#        [0.01, 0.07, 0.92]]
#
# cell = [[0.87, 0.04, 0.10],
#         [0.05, 0.80, 0.15],
#         [0.03, 0.05, 0.92]]
#
# ori = [[0.78, 0.13, 0.10],
#        [0.08, 0.86, 0.06],
#        [0.02, 0.04, 0.94]]


save_to = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/figures/fib.png"
plot_confusion_matrix(cm=np.array(fib), save_to=save_to,
                      normalize=True,
                      target_names=['0', '1', '2'],
                      title="Fibrosis")

save_to = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/figures/cel.png"
plot_confusion_matrix(cm=np.array(cell),  save_to=save_to,
                      normalize=True,
                      target_names=['0', '1', '2'],
                      title="Cellularity")

save_to = "/Jun_anonymized_dir/OvaryCancer/StromaReaction/figures/par.png"
plot_confusion_matrix(cm=np.array(ori),  save_to=save_to,
                      normalize=True,
                      target_names=['0', '1', '2'],
                      title="Orientation")

print("Done")

