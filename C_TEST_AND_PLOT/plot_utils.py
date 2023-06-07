import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager

from matplotlib import rcParams
matplotlib.rcParams['font.sans-serif'] = "Manjari"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"
plt.rc('axes', axisbelow=True)
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
#print(matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf'))


def create_colors():

    colors = {}
    colors['green'] = '#3D8361'
    colors['dark_green'] = '#1C6758'
    colors['magenta'] = '#7D1935'
    colors['dark_blue'] = '#002B5B'
    colors['black'] = '#000000'
    colors['white'] = '#ffffff'
    colors['light_grey'] = '#D8D8D8'
    colors['dark_grey'] = '#393E46'

    colors['green1'] = '#539575'
    colors['green2'] = '#3D8361'
    colors['green3'] = '#04724D'
    colors['green4'] = '#1C6758'

    colors['exp1a'] = '#FC7300'
    colors['exp1b'] = '#00425A'
    colors['exp1c'] = '#1F8A70'
    colors['exp1d'] = '#BFDB38'


    return colors

def bar_plot_metrics(d, sequences_results, tot_intervals_results, start_intervals_results, metrics, version, model):

    other_info = '3Bars'

    if model == 'cnn_lstm_mm':
        modality = 'face+pose'
    elif model == 'cnn_lstm_image':
        modality = 'only-face'
    elif model == 'cnn_lstm_pose':
        modality = 'only-pose'
    elif model == 'cnn_lstm_mm_latefusion':
        modality = 'face+pose_latefusion'

    if version == 'v18_involved' or version == 'vTHESIS_50_binary' or version == 'vTHESIS_50_512_binary':
        classification = 'binary class. ADDRESSED vs NOT ADDRESSED'
    elif version == 'v16_nao_left_right' or version == 'v_20_nao_left_right_softmaxrelu' or version == 'v_21_nao_left_right_softmaxrelu' \
            or version == 'v_21_nao_left_right_leakyrelu' or version == 'v_22_nao_left_right_softmaxrelu' \
            or version  == 'vTHESIS_50'  or version  == 'vTHESIS_50_b'  or version  == 'vTHESIS_50_512' or version  == 'vTHESIS_50_512b'\
            or version  == 'vTHESIS_50_64':
        classification = 'three classes addressee localization'

    colors = create_colors()

    if metrics == 'weighted_avg_f1score':
        title = 'WEIGHTED F1 SCORE'
        save_title = 'F1SCORE'
    elif metrics == 'weighted_recall':
        title = 'WEIGHTED RECALL'
        save_title = 'RECALL'
    elif metrics == 'weighted_precision':
        title = 'WEIGHTED PRECISION'
        save_title = 'PRECISION'
    elif metrics == 'accuracy':
        title = 'ACCURACY'
        save_title = 'ACCURACY'

    data = np.array([])
    data_std = np.array([])
    all_results = [sequences_results, tot_intervals_results, start_intervals_results]

    for df in all_results:
        #print("results", round(df[metrics].mean(),2))
        data = np.append(data, round(df[metrics].mean(), 2))
        data_std = np.append(data_std, round(df[metrics].std(), 2))
    #print(data)

    main_index = np.arange(3) + 1
    y_ticks = (np.arange(6)) * 20
    main_labels = ['sequences\n(0.8 s)', 'utterances', 'utterances \n(first 0.8 s)']
    bar_width = 0.4

    plt.grid(color=colors['light_grey'], linestyle='-', linewidth=1, axis='y')
    plt.bar(main_index, data, bar_width, color=colors['green'])
    plt.errorbar(main_index, data, yerr=data_std, label='both limits (default)', ls='none', color=colors['dark_grey'])
    for s, score in enumerate(data):
        plt.text(main_index[s]-0.18, score-60, '{} %'.format(score), fontsize=14, color=colors['white'])

    plt.subplots_adjust(bottom=0.15, top=0.8)
    ax = plt.gca()
    ax.set_xticks(main_index)
    ax.set_xticklabels(main_labels, fontsize=16)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(['0', '20', '40', '60', '80', '100'], fontsize=14)
    plt.ylim(top=100)
    plt.yticks()
    plt.ylabel('%')
    #plt.title('{}\nin {} ({})\n 10-folds cross validation'.format(title, classification, modality), fontsize=16)
    #plt.savefig(os.path.join(d['save_plot_dir'], 'barplot_{}_{}_{}_{}.png'.format(other_info, save_title, version, modality)), dpi=300)
    plt.show()

    return


def bar_plot_classes(d, tot_intervals_results, version, model):

    other_info = 'class_performance'
    colors = create_colors()
    if version == 'v18_involved' or version  == 'vTHESIS_50_binary' or version == 'vTHESIS_50_512_binary':
        classification = 'binary class. ADDRESSED vs NOT ADDRESSED'
        main_labels = ['ADDRESSED', 'NOT_ADDRESSED']
    elif version == 'v16_nao_left_right' or version == 'v_20_nao_left_right_softmaxrelu' or version == 'v_21_nao_left_right_softmaxrelu' \
            or version == 'v_21_nao_left_right_leakyrelu' or version == 'v_22_nao_left_right_softmaxrelu' \
            or version  == 'vTHESIS_50' or version  == 'vTHESIS_50_512' or version  == 'vTHESIS_50_b' or version  == 'vTHESIS_50_512b'\
            or version  == 'vTHESIS_50_64':
        classification = '3-classes addressee localization'
        main_labels = ['LEFT', 'ROBOT', 'RIGHT']

    if model == 'cnn_lstm_mm':
        modality = 'face+pose'
    elif model == 'cnn_lstm_image':
        modality = 'only-face'
    elif model == 'cnn_lstm_pose':
        modality = 'only-pose'
    elif model == 'cnn_lstm_mm_latefusion':
        modality = 'face+pose_latefusion'

    data_recall = np.array([])
    data_f1score = np.array([])
    data_precision = np.array([])
    data_recall_std = np.array([])
    data_f1score_std = np.array([])
    data_precision_std = np.array([])

    df = tot_intervals_results
    for l, label in enumerate(main_labels):

        data_recall = np.append(data_recall, round(df['{}_recall'.format(label)].mean(), 2))
        data_f1score = np.append(data_f1score, round(df['{}_f1score'.format(label)].mean(), 2))
        data_precision = np.append(data_precision, round(df['{}_precision'.format(label)].mean(), 2))
        data_recall_std = np.append(data_recall_std, round(df['{}_recall'.format(label)].std()/2, 2))
        data_f1score_std = np.append(data_f1score_std, round(df['{}_f1score'.format(label)].std()/2, 2))
        data_precision_std = np.append(data_precision_std, round(df['{}_precision'.format(label)].std()/2, 2))

    main_index = np.arange(3) + 1
    minor_ticks = np.array([0.7, 1, 1.3, 1.7, 2, 2.3, 2.7, 3, 3.3])
    x_labels = ['recall', 'precision', 'LEFT', 'f1 score', 'recall', 'precision', 'ROBOT', 'f1 score', 'recall', 'precision', 'RIGHT', 'f1 score']
    y_ticks = (np.arange(6)) * 20

    bar_width = 0.3

    plt.grid(color=colors['light_grey'], linestyle='-', linewidth=1, axis='y')
    bar_recall = plt.bar(main_index - 0.3, data_recall, bar_width, color=colors['dark_blue'])
    bar_precision = plt.bar(main_index, data_precision, bar_width, color=colors['magenta'])
    bar_f1score = plt.bar(main_index + 0.3, data_f1score, bar_width, color=colors['green'])
    plt.errorbar(main_index - 0.3, data_recall, yerr=data_recall_std, label='both limits (default)', ls='none', color=colors['dark_grey'])
    plt.errorbar(main_index, data_precision, yerr=data_precision_std, label='both limits (default)', ls='none', color=colors['dark_grey'])
    plt.errorbar(main_index + 0.3, data_f1score, yerr=data_f1score_std, label='both limits (default)', ls='none', color=colors['dark_grey'])

    #for s, _ in enumerate(data_recall):
    #    plt.text(main_index[s] - 0.42, data_recall[s]-55, '{}\n   %'.format(data_recall[s]), fontsize=12, color=colors['white'])
    #    plt.text(main_index[s] - 0.12, data_precision[s]-55, '{}\n   %'.format(data_precision[s]), fontsize=12, color=colors['white'])
    #    plt.text(main_index[s] + 0.18, data_f1score[s]-55, '{}\n   %'.format(data_f1score[s]), fontsize=12, color=colors['white'])

    plt.subplots_adjust(bottom=0.1, top=0.8)
    ax = plt.gca()
    ax.set_xticks(main_index)
    #ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticklabels(main_labels, fontsize=16)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(['0', '20', '40', '60', '80', '100'], fontsize=14)
    plt.ylim(top=100)
    plt.yticks()
    plt.ylabel('%')
    plt.legend([bar_recall, bar_precision, bar_f1score], ['recall', 'precision', 'f1 score'], ncol=3)
    #plt.title('CLASS PERFORMANCE\nin {} ({})\n 10-folds cross validation'.format(classification, modality), fontsize=16)
    #plt.savefig(os.path.join(d['save_plot_dir'], 'barplot_{}_{}_{}.png'.format(other_info, version, modality)), dpi=300)
    plt.show()

    return

def bar_plot_binary(d, sequences_results, version, model):

    other_info = 'binary_performance'
    colors = create_colors()
    classification = 'binary class. ADDRESSED vs NOT ADDRESSED'

    if model == 'cnn_lstm_mm':
        modality = 'face+pose'
    elif model == 'cnn_lstm_image':
        modality = 'only-face'
    elif model == 'cnn_lstm_pose':
        modality = 'only-pose'
    elif model == 'cnn_lstm_mm_latefusion':
        modality = 'face+pose_latefusion'

    data_recall = np.array([])
    data_f1score = np.array([])
    data_precision = np.array([])
    data_recall_std = np.array([])
    data_f1score_std = np.array([])
    data_precision_std = np.array([])

    df = sequences_results
    for l, label in enumerate(['ADDRESSED', 'NOT_ADDRESSED']):

        data_recall = np.append(data_recall, round(df['{}_recall'.format(label)].mean(), 2))
        data_f1score = np.append(data_f1score, round(df['{}_f1score'.format(label)].mean(), 2))
        data_precision = np.append(data_precision, round(df['{}_precision'.format(label)].mean(), 2))
        data_recall_std = np.append(data_recall_std, round(df['{}_recall'.format(label)].std(), 2))
        data_f1score_std = np.append(data_f1score_std, round(df['{}_f1score'.format(label)].std(), 2))
        data_precision_std = np.append(data_precision_std, round(df['{}_precision'.format(label)].std(), 2))

    main_index = np.arange(4) + 1

    y_ticks = (np.arange(6)) * 20
    main_labels = ['sensibility', 'precision', 'F1score', 'sensitivity']
    bar_width = 0.6

    plt.grid(color=colors['light_grey'], linestyle='-', linewidth=1, axis='y')
    bar_recall = plt.bar(main_index[0], data_recall[0], bar_width, color=colors['dark_blue'])
    bar_precision = plt.bar(main_index[1], data_precision[0], bar_width, color=colors['magenta'])
    bar_f1score = plt.bar(main_index[2], data_f1score[0], bar_width, color=colors['green'])
    bar_sensitivity = plt.bar(main_index[3], data_recall[1], bar_width, color=colors['dark_blue'])
    plt.errorbar(main_index[0], data_recall[0], yerr=data_recall_std[0], label='both limits (default)', ls='none',
                 color=colors['dark_grey'])
    plt.errorbar(main_index[1], data_precision[0], yerr=data_precision_std[0], label='both limits (default)', ls='none',
                 color=colors['dark_grey'])
    plt.errorbar(main_index[2], data_f1score[0], yerr=data_f1score_std[0], label='both limits (default)', ls='none',
                 color=colors['dark_grey'])
    plt.errorbar(main_index[3], data_recall[1], yerr=data_recall_std[1], label='both limits (default)', ls='none',
                 color=colors['dark_grey'])

    plt.text(main_index[0]-0.27, data_recall[0]-60, '{} %'.format(data_recall[0]), fontsize=14,
             color=colors['white'])
    plt.text(main_index[1]-0.27, data_precision[0]-60, '{} %'.format(data_precision[0]), fontsize=14,
             color=colors['white'])
    plt.text(main_index[2]-0.27, data_f1score[0]-60, '{} %'.format(data_f1score[0]), fontsize=14,
             color=colors['white'])
    plt.text(main_index[3]-0.27, data_recall[1] - 60, '{} %'.format(data_recall[1]), fontsize=14,
             color=colors['white'])

    plt.subplots_adjust(bottom=0.1, top=0.8)
    ax = plt.gca()
    ax.set_xticks(main_index)
    #ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticklabels(main_labels, fontsize=16)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(['0', '20', '40', '60', '80', '100'], fontsize=14)
    plt.ylim(top=100)
    plt.yticks()
    plt.ylabel('%')
    #plt.legend([bar_recall, bar_precision, bar_f1score, bar_sensitivity], ['recall', 'precision', 'f1 score', 'sensitivity'], ncol=4)
    #plt.title('CLASS PERFORMANCE\nin {} ({})\n 10-folds cross validation'.format(classification, modality), fontsize=16)
    #plt.savefig(os.path.join(d['save_plot_dir'], 'barplot_binary_{}_{}_{}.png'.format(other_info, version, modality)), dpi=300)
    plt.show()

    return


def bar_plot_gradual_accuracy(d, gradual_intervals_results):

    data = np.array([])
    data_std = np.array([])

    for duration in ['accuracy_1', 'accuracy_2', 'accuracy_3']:
        data = np.append(data, round(gradual_intervals_results[duration].mean(), 2))
        data_std = np.append(data_std, round(gradual_intervals_results[duration].std(), 2))

    main_index = np.arange(3) + 1
    main_labels = ['utterances = 0.8 s', 'utterance = 1.6 s', 'utterances >= 2.4 s']
    bar_width = 0.4
    color = [0.1, 0.3, 0.7]

    plt.bar(main_index, data, bar_width, color=color)
    plt.errorbar(main_index, data, yerr=data_std, label='both limits (default)', ls='none', color=[0, 0, 0])
    for s, score in enumerate(data):
        plt.text(main_index[s] - 0.2, 5, '{} %'.format(score), fontsize=12, color=[1, 1, 1])

    plt.subplots_adjust(left=0.2, bottom=0.2)
    ax = plt.gca()
    ax.set_xticks(main_index)
    ax.set_xticklabels(main_labels)
    plt.ylim(top=100)
    plt.yticks()
    plt.ylabel('%')
    #plt.title('GRADUAL ACCURACY \n 10-folds cross validation')
    plt.show()

    return


def bar_plot_incremental_accuracy(d, incremental_intervals_results, metrics, version, model):

    other_info = 'incrementalBars'
    colors = create_colors()
    data = np.array([])
    data_std = np.array([])

    if model == 'cnn_lstm_mm':
        modality = 'face+pose'
    elif model == 'cnn_lstm_image':
        modality = 'only-face'
    elif model == 'cnn_lstm_pose':
        modality = 'only-pose'
    elif model == 'cnn_lstm_mm_latefusion':
        modality = 'face+pose_latefusion'

    if version == 'v18_involved' or version == 'vTHESIS_50_binary' or version == 'vTHESIS_50_512_binary':
        classification = 'binary class. ADDRESSED vs NOT ADDRESSED'
    elif version == 'v16_nao_left_right' or version == 'v_20_nao_left_right_softmaxrelu' or version == 'v_21_nao_left_right_softmaxrelu' \
            or version == 'v_21_nao_left_right_leakyrelu' or version == 'v_22_nao_left_right_softmaxrelu' \
            or version  == 'vTHESIS_50'  or version  == 'vTHESIS_50_b'  or version  == 'vTHESIS_50_512' or version  == 'vTHESIS_50_512b'\
            or version  == 'vTHESIS_50_64':
        classification = 'three classes addressee localization'

    if metrics == 'weighted_avg_f1score':
        title = 'WEIGHTED F1 SCORE'
        save_title = 'F1SCORE'
        parameters = ['weighted_avg_f1score_1', 'weighted_avg_f1score_2', 'weighted_avg_f1score_3',
                      'weighted_avg_f1score_4']
    elif metrics == 'weighted_recall':
        title = 'WEIGHTED RECALL'
        save_title = 'RECALL'
        parameters = ['weighted_recall_1', 'weighted_recall_2', 'weighted_recall_3',
                      'weighted_recall_4']
    elif metrics == 'weighted_precision':
        title = 'WEIGHTED PRECISION'
        save_title = 'PRECISION'
        parameters = ['weighted_precision_1', 'weighted_precision_2', 'weighted_precision_3',
                      'weighted_precision_4']
    elif metrics == 'accuracy':
        title = 'ACCURACY'
        save_title = 'ACCURACY'
        parameters = ['accuracy_1', 'accuracy_2', 'accuracy_3', 'accuracy_4']

    for parameter in parameters:

        data = np.append(data, round(incremental_intervals_results[parameter].mean(), 2))
        data_std = np.append(data_std, round(incremental_intervals_results[parameter].std(), 2))

    main_index = np.arange(4) + 1
    y_ticks = (np.arange(6))*20
    main_labels = ['prediction\nat 0.8 s', 'prediction\nat 1.6 s', 'prediction\nat 2.4 s', 'prediction\n>= 2.4 s']
    bar_width = 0.6
    incremental_colors = [colors['green1'], colors['green2'], colors['green3'], colors['green4']]
    plt.grid(color=colors['light_grey'], linestyle='-', linewidth=1, axis='y')
    plt.bar(main_index, data, bar_width, color=incremental_colors)
    plt.errorbar(main_index, data, yerr=data_std, label='both limits (default)', ls='none', color=colors['dark_grey'])
    for s, score in enumerate(data):
        plt.text(main_index[s] - 0.25, score-70, '{}%'.format(score), fontsize=14, color=colors['white'])

    plt.subplots_adjust(bottom=0.15, top=0.8)
    ax = plt.gca()
    ax.set_xticks(main_index)
    ax.set_xticklabels(main_labels, fontsize=14)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(['0', '20', '40', '60', '80', '100'], fontsize=14)
    plt.ylim(top=100)
    plt.yticks()
    plt.ylabel('%')
    #plt.title('{} with INCREMENTAL DURATION\nin {} ({})\n10-folds cross validation'.format(title, classification, modality), fontsize=16)
    #plt.savefig(os.path.join(d['save_plot_dir'], 'barplot_{}_{}_{}_{}.png'.format(other_info, save_title, version, modality)),
    #            dpi=300)
    plt.show()

    return


def plot_confusion_matrix_(d, performance_test, icub_ds):

    if d['model'] == 'cnn_lstm_mm':
        modality = 'face+pose'
    elif d['model'] == 'cnn_lstm_image':
        modality = 'only-face'
    elif d['model'] == 'cnn_lstm_pose':
        modality = 'only-pose'
    elif d['model'] == 'cnn_lstm_mm_latefusion':
        modality = 'face+pose_latefusion'


    fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(nrows=2, ncols=2, gridspec_kw={'width_ratios':[3,1], 'height_ratios':[3,1]})
    plt.subplots_adjust(bottom=0.1, top=0.8, right=0.65)
    # plot 1 confusion matrix
    im = ax1.imshow(d['confusion_matrix'], cmap='Greens', vmin=0, vmax=119)  # x axis = real class, y axis = predicted class

    ax1.set_xticks(np.arange(len(d['labels'])))
    ax1.set_yticks(np.arange(len(d['labels'])))
    ax1.tick_params(labelsize=14)
    ax1.set_xticklabels(d['labels'])
    ax1.set_yticklabels(d['labels'])
    ax1.set_ylabel('Real Class', fontweight='bold', fontsize=18)
    ax1.xaxis.set_label_coords(.5, 1.15)
    ax1.xaxis.tick_top()
    ax1.set_xlabel('Predicted Class', fontweight='bold', fontsize=18)
    ax1.xaxis.set_label_position('top')
    ax1.tick_params(axis='y', which='both', rotation=70)
    #ax1.set_title("CONFUSION MATRIX {}".format(modality))

    # Loop over data dimensions and create text annotations.
    for i in range(len(d['labels'])):
        for j in range(len(d['labels'])):
            text = ax1.text(j, i, int(d['confusion_matrix'][i, j]),
                           ha="center", va="center", color="k", fontsize=18)

    ## plot 2 recall
    ax2.set_title("Recall\n(%)", fontweight ='bold', fontsize=18)
    im = ax2.imshow(performance_test['recall'], cmap='Greens', vmin=0, vmax=100)
    for i, c in enumerate(d['labels']):
        text = ax2.text(0, i, str(round(performance_test['recall'].iloc[i][0], 2)), ha="center", va="center", color="k", fontsize=18)
    ax2.tick_params(top=False, bottom=False, left=False, right=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)

    ## plot 3 precision
    ax3.set_xlabel('Precision (%)', fontweight='bold', fontsize=18)
    im = ax3.imshow(performance_test['precision'].T, cmap='Greens', vmin=0, vmax=100)
    for i, c in enumerate(d['labels']):
        text = ax3.text(i, 0, str(round(performance_test['precision'].iloc[i][0], 2)), ha="center", va="center",
                        color="k", fontsize=18)
    ax3.tick_params(top=False, bottom=False, left=False, right=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)

    df = pd.DataFrame([])
    df.loc[0, 'a'] = 2
    df.loc[0, 'b'] = 2

    performance_test['overall_acc'].iloc[0][0] = icub_ds['weighted_avg_f1score']
    # plot 4
    #ax4.xaxis.set_label_position("bottom")
    #ax4.xaxis.tick_right()
    ax4.set_xlabel('F1score (%)', fontweight='bold', fontsize=18)
    #ax4.set_title("%", fontweight ='bold', fontsize=16)
    im = ax4.imshow(performance_test['overall_acc'], cmap='Greens', vmin=0, vmax=100)
    text = ax4.text(0, 0, str(round(performance_test['overall_acc'].iloc[0][0], 2)), ha="center", va="center",
                    color="k", fontsize=18)

    ax4.tick_params(top=False, bottom=False, left=False, right=False)
    plt.setp(ax4.get_xticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)
    #ax4.set_yticks(np.arange(2))
    #ax4.set_yticklabels(performance_test['overall_acc'].index.values)
    #ax4.set_yticklabels(['accuracy', 'error rate'])
    save_dir = os.path.join(d['save_plot_dir'], 'conf_mtrx_{}.png'.format(modality))
    #plt.savefig(save_dir, dpi=300)
    plt.show()

    return


def plot_comparison(d, three_classes_df, binary_df, metrics):

    colors = create_colors()
    other_info = 'comparison'
    modalities = ['cnn_lstm_image', 'cnn_lstm_pose', 'cnn_lstm_mm']
    modalities_dict = {'cnn_lstm_image': 'only_face', 'cnn_lstm_pose': 'only_pose', 'cnn_lstm_mm': 'face+pose'}

    if metrics == 'weighted_avg_f1score':
        title = 'WEIGHTED F1 SCORE'
        save_title = 'F1SCORE'

    #data_face = np.array([])
    #data_pose = np.array([])
    data_mm = np.array([])
    #data_face_std = np.array([])
    #data_pose_std = np.array([])
    data_mm_std = np.array([])

    for c, classification in enumerate(['vTHESIS_50_binary', 'vTHESIS_50_512_b']):

        if classification == 'vTHESIS_50_512_b':
            df = three_classes_df
        elif classification == 'vTHESIS_50_binary':
            df = binary_df
        #data_face = np.append(data_face, round(df['{}_{}'.format('weighted_avg_f1score', 'cnn_lstm_image')].mean(), 2))
        #data_pose = np.append(data_pose, round(df['{}_{}'.format('weighted_avg_f1score', 'cnn_lstm_pose')].mean(), 2))
        data_mm = np.append(data_mm, round(df['{}_{}'.format('weighted_avg_f1score', 'cnn_lstm_mm')].mean(), 2))
        #data_face_std = np.append(data_face_std, round(df['{}_{}'.format('weighted_avg_f1score', 'cnn_lstm_image')].std(), 2))
        #data_pose_std = np.append(data_pose_std, round(df['{}_{}'.format('weighted_avg_f1score', 'cnn_lstm_pose')].std(), 2))
        data_mm_std = np.append(data_mm_std, round(df['{}_{}'.format('weighted_avg_f1score', 'cnn_lstm_mm')].std(), 2))

    main_index = np.arange(2) + 1
    y_ticks = (np.arange(6)) * 20
    main_labels = ['BINARY\nADDRESSED VS NOT', 'THREE CLASSES\nLOCALIZATION']
    bar_width = 0.3

    plt.grid(color=colors['light_grey'], linestyle='-', linewidth=1, axis='y')
#    bar_face = plt.bar(main_index - 0.3, data_face, bar_width, color=colors['dark_blue'])
#    bar_pose = plt.bar(main_index, data_pose, bar_width, color=colors['magenta'])
    bar_mm = plt.bar(main_index + 0.3, data_mm, bar_width, color=colors['green'])
#    plt.errorbar(main_index - 0.3, data_face, yerr=data_face_std, label='both limits (default)', ls='none',
#                 color=colors['dark_grey'])
#    plt.errorbar(main_index, data_pose, yerr=data_pose_std, label='both limits (default)', ls='none',
#                 color=colors['dark_grey'])
    plt.errorbar(main_index + 0.3, data_mm, yerr=data_mm_std, label='both limits (default)', ls='none',
                 color=colors['dark_grey'])

    for s, _ in enumerate(data_mm):
        #plt.text(main_index[s] - 0.42, data_face[s]-55, '{}\n   %'.format(data_face[s]), fontsize=16, color=colors['white'])
        #plt.text(main_index[s] - 0.12, data_pose[s]-55, '{}\n   %'.format(data_pose[s]), fontsize=16, color=colors['white'])
        plt.text(main_index[s] + 0.2, data_mm[s]-55, '{}\n   %'.format(data_mm[s]), fontsize=16, color=colors['white'])

    plt.subplots_adjust(bottom=0.13, top=0.8)
    ax = plt.gca()
    ax.set_xticks(main_index)
    ax.set_xticklabels(main_labels, fontsize=16)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(['0', '20', '40', '60', '80', '100'], fontsize=14)
    plt.ylim(top=100)
    plt.yticks()
    plt.ylabel('%')
    #plt.legend([bar_face, bar_pose, bar_mm], ['only face', 'only pose', 'face+pose'], ncol=3)
    plt.legend([bar_mm], ['face+pose'], ncol=1)
    plt.title('COMPARISON AMONG MODALITIES\nin {}\n 10-folds cross validation'.format(title), fontsize=16)
    #plt.savefig(os.path.join(d['save_plot_dir'], 'barplot_{}_{}.png'.format(other_info, 'f1score')), dpi=300)
    plt.show()


    return



def trend_incremental_accuracy(d, model):

    other_info = 'incrementalTrend'
    colors = create_colors()
    data = np.array([])
    data_std = np.array([])

    if model == 'cnn_lstm_mm':
        modality = 'face+pose'
    elif model == 'cnn_lstm_image':
        modality = 'only-face'
    elif model == 'cnn_lstm_pose':
        modality = 'only-pose'
    elif model == 'cnn_lstm_mm_latefusion':
        modality = 'face+pose_latefusion'


    title = 'WEIGHTED F1 SCORE'
    save_title = 'F1SCORE'
    parameters = ['weighted_avg_f1score_1', 'weighted_avg_f1score_2', 'weighted_avg_f1score_3',
                      'weighted_avg_f1score_4']

    exp1a = [74.15, 76.48, 76.5, 79.8]
    exp1b = [71.88, 72.03, 75.22, 77.22]
    exp1c = [72.07, 75.04, 77.26, 78.25]
    exp1d = [70.77, 70.23, 70.44, 70.49]
    main_index = np.arange(4)
    y_ticks = (np.arange(11))*10
    main_labels = ['prediction\nat 0.8 s', 'prediction\nat 1.6 s', 'prediction\nat 2.4 s', 'prediction\n>= 2.4 s']
    line1 = plt.plot(exp1a, linestyle='dashdot', color=colors['exp1a'], linewidth='3', label='Exp. 1.a')
    line2 = plt.plot(exp1b, linestyle='dashed', color=colors['exp1b'], linewidth='3', label='Exp. 1.b')
    line3 = plt.plot(exp1c, linestyle='dotted', color=colors['exp1c'], linewidth='3', label='Exp. 1.c')
    line4 = plt.plot(exp1d, linestyle='solid', color=colors['exp1d'], linewidth='3', label='Exp. 1.d')
    line1 = plt.plot(exp1a, linestyle='dashdot', color=colors['exp1a'], linewidth='3')

    plt.grid(color=colors['light_grey'], linestyle='-', linewidth=1, axis='both')
    #plt.bar(main_index, data, bar_width, color=incremental_colors)
    #plt.errorbar(main_index, data, yerr=data_std, label='both limits (default)', ls='none', color=colors['dark_grey'])
    #for s, score in enumerate(data):
    #    plt.text(main_index[s] - 0.25, score-70, '{}%'.format(score), fontsize=14, color=colors['white'])

    plt.subplots_adjust(bottom=0.15, top=0.8)
    ax = plt.gca()
    ax.set_xticks(main_index)
    ax.set_xticklabels(main_labels, fontsize=14)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'], fontsize=14)
    plt.ylim(bottom=65, top=85)
    plt.yticks()
    plt.ylabel('%')
    ax.legend(loc='lower left')

    #plt.title('{} with INCREMENTAL DURATION\nin {} ({})\n10-folds cross validation'.format(title, classification, modality), fontsize=16)
    #plt.savefig(os.path.join(d['save_plot_dir'], 'trendplot_{}_{}.png'.format(other_info, save_title)),
    #            dpi=300)
    plt.show()

    return


def bar_IJCNN(d):

    other_info = 'bar plot'

    colors = create_colors()

    exp1a_avg = [75.01, 76.48, 74.15]
    exp1b_avg = [73.18, 74.19, 71.88]
    exp1c_avg = [72.83, 73.22, 72.07]
    exp1d_avg = [72.6, 71.05, 70.77]

    exp1a_std = [8.6, 8.42, 9.19]
    exp1b_std = [7.57, 7.97, 5.03]
    exp1c_std = [5.86, 6.93, 8.89]
    exp1d_std = [6.75, 7.76, 8.84]

    tests_list = ['sequences', 'utterances', 'first sequence']
    for n, test in enumerate(tests_list):
        print(n, tests_list.index(test))
        data = [exp1a_avg[n], exp1b_avg[n], exp1c_avg[n], exp1d_avg[n]]
        data_std = [exp1a_std[n]/2, exp1b_std[n]/2, exp1c_std[n]/2, exp1d_std[n]/2]
        c = [colors['exp1a'], colors['exp1b'], colors['exp1c'], colors['exp1d']]
        c_text = [colors['black'], colors['black'], colors['black'], colors['black']]
        main_index = np.arange(4) + 1
        y_ticks = (np.arange(6)) * 20
        main_labels = ['Exp. 1.a', 'Exp. 1.b', 'Exp. 1.c', 'Exp. 1.d']
        bar_width = 0.7

        plt.grid(color=colors['light_grey'], linestyle='-', linewidth=1, axis='y')
        plt.bar(main_index, data, bar_width, color=c)
        plt.errorbar(main_index, data, yerr=data_std, label='both limits (default)', ls='none', color=colors['dark_grey'])
        for s, score in enumerate(data):
            plt.text(main_index[s]-0.35, score-60, '{} %'.format(score), fontsize=18, color=c_text[s])

        plt.subplots_adjust(bottom=0.15, top=0.8)
        ax = plt.gca()
        ax.set_xticks(main_index)
        ax.set_xticklabels(main_labels, fontsize=16)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(['0', '20', '40', '60', '80', '100'], fontsize=14)
        plt.ylim(top=100)
        plt.yticks()
        plt.ylabel('%')
        plt.title('{} of F1 score in {}\n (10-folds cross validation)'.format(other_info, tests_list[n]), fontsize=16)
        #plt.savefig(os.path.join(d['save_plot_dir'], 'barplot_{}_{}.png'.format(other_info, tests_list[n])), dpi=300)
        plt.show()

    return