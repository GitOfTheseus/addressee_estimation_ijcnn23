import matplotlib.pyplot as plt
import numpy as np
import os

np.random.seed(0)

def plot_performances(epoch, performances, d):

    epochs = range(1, epoch + 1)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.xaxis.set_label_coords(0.1, -0.2)
    ax1.plot(epochs, performances['train_loss'], 'g', label='Training loss')
    ax1.plot(epochs, performances['eval_loss'], 'b', label='Validation loss')
    ax1.set_title('Training and Validation loss')
    ax1.set(xlabel='Epochs', ylabel='Loss')
    ax1.xaxis.set_label_coords(0.1, -0.2)
    ax1.legend()
    ax2.plot(epochs, performances['train_acc'], 'g', label='Training accuracy')
    ax2.plot(epochs, performances['eval_acc'], 'b', label='Validation accuracy')
    ax2.set_title('Training and Validation accuracy')
    ax2.set(xlabel='Epochs', ylabel='Accuracy')
    ax2.xaxis.set_label_coords(0.1, -0.2)
    ax2.legend()

    plt.tight_layout()

    fig_dir = os.path.join(d['plot_dir'], d['model_name'] + '_perf_training.png')
    plt.savefig(fig_dir)
    # This will clear the first plot
    plt.close('all')

    return


def plot_confusion_matrix(d, performance_test):

    fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(nrows=2, ncols=2, gridspec_kw={'width_ratios':[4,1], 'height_ratios':[4,1]})

    # plot 1 confusion matrix
    im = ax1.imshow(d['confusion_matrix'])  # x axis = real class, y axis = predicted class
    ax1.set_xticks(np.arange(len(d['class_names'])))
    ax1.set_yticks(np.arange(len(d['class_names'])))
    ax1.set_xticklabels(d['class_names'])
    ax1.set_yticklabels(d['class_names'])
    ax1.set_ylabel('Real Class', fontweight='bold', fontsize=9)
    ax1.xaxis.set_label_coords(.5, 1.15)
    ax1.xaxis.tick_top()
    ax1.set_xlabel('Predicted Class', fontweight='bold', fontsize=9)
    ax1.xaxis.set_label_position('top')
    ax1.tick_params(axis='y', which='both', rotation=90)

    # Loop over data dimensions and create text annotations.
    for i in range(len(d['class_names'])):
        for j in range(len(d['class_names'])):
            text = ax1.text(j, i, d['confusion_matrix'][i, j],
                           ha="center", va="center", color="w")

    ## plot 2 recall
    ax2.set_title("Recall", fontweight ='bold', fontsize=12)
    im = ax2.imshow(performance_test['recall'])
    for i, c in enumerate(d['class_names']):
        text = ax2.text(0, i, str(round(performance_test['recall'].iloc[i][0], 2)), ha="center", va="center", color="w")
    ax2.tick_params(top=False, bottom=False, left=False, right=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)

    ## plot 3 precision
    ax3.set_title("Precision", fontweight ='bold', fontsize=12)
    im = ax3.imshow(performance_test['precision'].T)
    for i, c in enumerate(d['class_names']):
        text = ax3.text(i, 0, str(round(performance_test['precision'].iloc[i][0], 2)), ha="center", va="center",
                        color="w")
    ax3.tick_params(top=False, bottom=False, left=False, right=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)

    # plot 4
    ax4.set_title("Overall Accuracy.", fontweight ='bold', fontsize=12)
    print(performance_test['overall_acc'])
    im = ax4.imshow(performance_test['overall_acc'])
    text = ax4.text(0, 0, str(round(performance_test['overall_acc'].iloc[0][0], 2)), ha="center", va="center",
                    color="w")

    ax4.tick_params(top=False, bottom=False, left=False, right=False)
    plt.setp(ax4.get_xticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)
    #ax4.set_yticks(np.arange(2))
    #ax4.set_yticklabels(performance_test['overall_acc'].index.values)
    #ax4.set_yticklabels(['accuracy', 'error rate'])

    fig_dir = os.path.join(d['plot_dir'], '{}_conf_mtrx.png'.format(d['model_name']))
    plt.savefig(fig_dir)
    plt.show()

    return