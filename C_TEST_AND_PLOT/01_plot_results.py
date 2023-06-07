import argparse
import os
import pandas as pd
import numpy as np
from C_TEST_AND_PLOT.plot_utils import bar_plot_metrics, bar_plot_classes, bar_plot_gradual_accuracy, \
    bar_plot_incremental_accuracy, plot_confusion_matrix_, plot_comparison, bar_plot_binary, \
    trend_incremental_accuracy, bar_IJCNN


def read_arguments(args):
    global d
    d = {
        'results_path': args.results_path,
        'model': args.model,
        'version': args.version,
        'results_file': args.results_file,
        'sequence_duration': args.sequence_duration,
        'save_plot_dir': args.save_plot_dir,
    }

    return


def read_variables():
    global d
    d['results_dir'] = os.path.join(d['results_path'], d['model'], d['version'])
    d['models_list'] = [m for m in os.listdir(d['results_dir']) if os.path.isdir(os.path.join(d['results_dir'], m))]
    d['models_list'].sort()

    d['labels_dict'] = {
        'vTHESIS_50': ['ROBOT', 'LEFT', 'RIGHT'],
        'vTHESIS_50_512b': ['ROBOT', 'LEFT', 'RIGHT'],
        'vTHESIS_50_binary': ['ADDRESSED', 'NOT_ADDRESSED'],
    }
    d['labels'] = d['labels_dict'][d['version']]

    return


def create_dataframes():
    empty = np.zeros(10)
    global sequences_results, tot_intervals_results, start_intervals_results, gradual_intervals_results, \
        incremental_intervals_results

    sequences_results = pd.DataFrame(columns=['empty'], data=empty)
    sequences_results['empty'] = empty
    tot_intervals_results = pd.DataFrame(columns=['empty'], data=empty)
    tot_intervals_results['empty']= empty
    start_intervals_results = pd.DataFrame(columns=['empty'], data=empty)
    start_intervals_results['empty'] = empty
    gradual_intervals_results = pd.DataFrame(columns=['empty'], data=empty)
    gradual_intervals_results['empty'] = empty
    gradual_intervals_results['empty'] = empty
    incremental_intervals_results = pd.DataFrame(columns=['empty'], data=empty)
    incremental_intervals_results['empty'] = empty
    incremental_intervals_results['empty'] = empty
    incremental_intervals_results['empty'] = empty
    incremental_intervals_results['empty'] = empty
    incremental_intervals_results['empty'] = empty
    incremental_intervals_results['empty'] = empty
    incremental_intervals_results['empty'] = empty

    for l, label in enumerate(d['labels']):

        sequences_results['{}_recall'.format(label)] = empty
        sequences_results['{}_precision'.format(label)] = empty
        sequences_results['{}_f1score'.format(label)] = empty
        tot_intervals_results['{}_recall'.format(label)] = empty
        tot_intervals_results['{}_precision'.format(label)] = empty
        tot_intervals_results['{}_f1score'.format(label)] = empty
        start_intervals_results['{}_recall'.format(label)] = empty
        start_intervals_results['{}_precision'.format(label)] = empty
        start_intervals_results['{}_f1score'.format(label)] = empty

        incremental_intervals_results['{}_recall_1'.format(label)] = empty
        incremental_intervals_results['{}_precision_1'.format(label)] = empty
        incremental_intervals_results['{}_f1score_1'.format(label)] = empty
        incremental_intervals_results['{}_recall_2'.format(label)] = empty
        incremental_intervals_results['{}_precision_2'.format(label)] = empty
        incremental_intervals_results['{}_f1score_2'.format(label)] = empty
        incremental_intervals_results['{}_recall_3'.format(label)] = empty
        incremental_intervals_results['{}_precision_3'.format(label)] = empty
        incremental_intervals_results['{}_f1score_3'.format(label)] = empty
        incremental_intervals_results['{}_recall_4'.format(label)] = empty
        incremental_intervals_results['{}_precision_4'.format(label)] = empty
        incremental_intervals_results['{}_f1score_4'.format(label)] = empty

    return


def create_dataframes_comparison():

    global binary_df, three_classes_df
    empty = np.zeros(10)
    if d['version'] == 'vTHESIS_50_binary':
        binary_df = pd.DataFrame(columns=['0'], data=empty)
        for model in ['cnn_lstm_mm']:
            binary_df['accuracy_{}'.format(model)] = empty
            binary_df['weighted_avg_f1score_{}'.format(model)] = empty
            print(d['version'])
    elif d['version'] == 'vTHESIS_50_512b':
        three_classes_df = pd.DataFrame(columns=['0'], data=empty)
        for model in ['cnn_lstm_mm']:
            three_classes_df['accuracy_{}'.format(model)] = empty
            three_classes_df['weighted_avg_f1score_{}'.format(model)] = empty
            d['version']


    return


def create_icub_dataframe():

    global icub_df
    empty = np.zeros(1)
    icub_df = pd.DataFrame(columns=['accuracy'], data=empty)
    icub_df['weighted_avg_f1score'] = empty
    icub_df['weighted_recall'] = empty
    icub_df['weighted_precision'] = empty

    for l, label in enumerate(d['labels']):

        icub_df['{}_recall'.format(label)] = empty
        icub_df['{}_precision'.format(label)] = empty
        icub_df['{}_f1score'.format(label)] = empty

    return

def compute_confusion_matrix(d, df):


    confusion_matrix = np.zeros((len(d['labels']), len(d['labels'])))
    for i, l in enumerate(df['label']):

        confusion_matrix[int(df.loc[i, 'label']), int(df.loc[i, 'prediction'])] += 1

    return confusion_matrix


def adjust_intervals(model_df):
    for i, interval in enumerate(model_df['interval']):
        if i > 0 and interval < model_df['interval'].loc[i - 1]:
            addition = model_df['interval'].loc[i - 1]
            model_df.loc[i:, 'interval'] = model_df.loc[i:, 'interval'] + addition
            break

    return model_df


def compute_performance(confusion_matrix):
    total_predicted = confusion_matrix.sum(0)
    total_actual = confusion_matrix.sum(1)
    totals = confusion_matrix.sum(1).sum(0)

    df = pd.DataFrame(confusion_matrix, index=d['labels'], columns=d['labels'])
    df_predictions = pd.DataFrame(total_predicted, index=d['labels'])
    df_actual = pd.DataFrame(total_actual, index=d['labels'])

    total_acc = 0
    recall = pd.DataFrame(np.zeros(len(d['labels'])), index=d['labels'])
    precision = pd.DataFrame(np.zeros(len(d['labels'])), index=d['labels'])
    overall_perf = pd.DataFrame(np.zeros(1), index=['accuracy'])
    for name in (d['labels']):
        total_acc += df.loc[name, name]
        recall.loc[name] = df.loc[name, name] / df_actual.loc[name] * 100
        precision.loc[name] = df.loc[name, name] / df_predictions.loc[name] * 100

    accuracy = total_acc / totals * 100
    error_rate = 100 - accuracy
    overall_perf.loc['accuracy', 0] = accuracy

    performance_test = {'accuracy': accuracy, 'error_rate': error_rate, 'recall': recall,
                        'precision': precision, 'overall_acc': overall_perf, 'n_actual_samples': df_actual}

    '''
    print('RESULTS OF TEST {}'.format(d['model_name']))
    print("the accuracy of the model is {} %".format(performance_test['accuracy']))
    print("the error rate of the model is {} %".format(performance_test['error_rate']))
    print("recall : \n{}".format(performance_test['recall']))
    print("precision : \n{}".format(performance_test['precision']))
    '''

    return performance_test


def f1_formula(recall, precision):
    try:
        f1 = 2 * (recall * precision) / (recall + precision)
    except:
        print("recall and precision are 0")
        f1 = 0

    return f1


def compute_f1(m, sequences_performance, tot_intervals_performance, start_intervals_performance, incremental_intervals_performances_list):

    sequences_f1scores_array = np.array([])
    tot_intervals_f1scores_array = np.array([])
    start_intervals_f1scores_array = np.array([])

    incremental_intervals_f1scores_array_1 = np.array([])
    incremental_intervals_f1scores_array_2 = np.array([])
    incremental_intervals_f1scores_array_3 = np.array([])
    incremental_intervals_f1scores_array_4 = np.array([])

    for l, label in enumerate(d['labels']):

        sequence_f1_score = f1_formula(sequences_results.loc[m, '{}_recall'.format(label)].item(),
                              sequences_results.loc[m, '{}_precision'.format(label)].item())
        sequences_results.loc[m, '{}_f1score'.format(label)] = sequence_f1_score
        sequences_f1scores_array = np.append(sequences_f1scores_array, sequence_f1_score)
        #print(tot_intervals_results)
        tot_intervals_f1_score = f1_formula(tot_intervals_results.loc[m, '{}_recall'.format(label)].item(),
                              tot_intervals_results.loc[m, '{}_precision'.format(label)].item())
        tot_intervals_results.loc[m, '{}_f1score'.format(label)] = tot_intervals_f1_score
        tot_intervals_f1scores_array = np.append(tot_intervals_f1scores_array, tot_intervals_f1_score)

        start_intervals_f1_score = f1_formula(start_intervals_results.loc[m, '{}_recall'.format(label)].item(),
                              start_intervals_results.loc[m, '{}_precision'.format(label)].item())
        start_intervals_results.loc[m, '{}_f1score'.format(label)] = start_intervals_f1_score
        start_intervals_f1scores_array = np.append(start_intervals_f1scores_array, start_intervals_f1_score)

        # INCREMENTAL
        incremental_f1_score_1 = f1_formula(incremental_intervals_results.loc[m, '{}_recall_1'.format(label)].item(),
                              incremental_intervals_results.loc[m, '{}_precision_1'.format(label)].item())
        incremental_intervals_results.loc[m, '{}_f1score_1'.format(label)] = incremental_f1_score_1
        incremental_intervals_f1scores_array_1 = np.append(incremental_intervals_f1scores_array_1, incremental_f1_score_1)

        incremental_f1_score_2 = f1_formula(incremental_intervals_results.loc[m, '{}_recall_2'.format(label)].item(),
                              incremental_intervals_results.loc[m, '{}_precision_2'.format(label)].item())
        incremental_intervals_results.loc[m, '{}_f1score_2'.format(label)] = incremental_f1_score_2
        incremental_intervals_f1scores_array_2 = np.append(incremental_intervals_f1scores_array_2, incremental_f1_score_2)

        incremental_f1_score_3 = f1_formula(incremental_intervals_results.loc[m, '{}_recall_3'.format(label)].item(),
                              incremental_intervals_results.loc[m, '{}_precision_3'.format(label)].item())

        incremental_intervals_results.loc[m, '{}_f1score_3'.format(label)] = incremental_f1_score_3
        incremental_intervals_f1scores_array_3 = np.append(incremental_intervals_f1scores_array_3, incremental_f1_score_3)

        incremental_f1_score_4 = f1_formula(incremental_intervals_results.loc[m, '{}_recall_4'.format(label)].item(),
                              incremental_intervals_results.loc[m, '{}_precision_4'.format(label)].item())
        incremental_intervals_results.loc[m, '{}_f1score_4'.format(label)] = incremental_f1_score_4
        incremental_intervals_f1scores_array_4 = np.append(incremental_intervals_f1scores_array_4, incremental_f1_score_4)

    sequences_results.loc[m, 'weighted_avg_f1score'] = compute_weighted_F1_average(sequences_performance, sequences_f1scores_array)
    tot_intervals_results.loc[m, 'weighted_avg_f1score'] = compute_weighted_F1_average(tot_intervals_performance, tot_intervals_f1scores_array)
    start_intervals_results.loc[m, 'weighted_avg_f1score'] = compute_weighted_F1_average(start_intervals_performance, start_intervals_f1scores_array)

    incremental_intervals_results.loc[m, 'weighted_avg_f1score_1'] = compute_weighted_F1_average(incremental_intervals_performances_list[0],
                                                                                                 incremental_intervals_f1scores_array_1)
    incremental_intervals_results.loc[m, 'weighted_avg_f1score_2'] = compute_weighted_F1_average(incremental_intervals_performances_list[1],
                                                                                                 incremental_intervals_f1scores_array_2)
    incremental_intervals_results.loc[m, 'weighted_avg_f1score_3'] = compute_weighted_F1_average(incremental_intervals_performances_list[2],
                                                                                                 incremental_intervals_f1scores_array_3)
    incremental_intervals_results.loc[m, 'weighted_avg_f1score_4'] = compute_weighted_F1_average(incremental_intervals_performances_list[3],
                                                                                                 incremental_intervals_f1scores_array_4)

    return


def compute_accuracy_sequence(model_df):
    n_samples = len(model_df)
    sequences_correct = sum(model_df['label'] == model_df['prediction'])
    sequences_accuracy = sequences_correct / n_samples * 100

    confusion_matrix = compute_confusion_matrix(d, model_df)
    performance = compute_performance(confusion_matrix)
    d['confusion_matrix'] = confusion_matrix

    return sequences_accuracy, performance


def compute_accuracy_tot_interval(model_df):
    # tot accuracy intervals
    intervals = np.sort(model_df['interval'].unique())
    intervals_labels = []
    intervals_predictions = []
    for i, interval in enumerate(intervals):
        interval_results = np.zeros(len(d['labels']))
        interval_label = model_df.loc[model_df['interval'] == interval, 'label'].iloc[0]
        predictions = model_df.loc[model_df['interval'] == interval, 'prediction'].tolist()
        scores = model_df.loc[model_df['interval'] == interval, 'score'].tolist()

        for p, prediction in enumerate(predictions):
            interval_results[prediction] = interval_results[prediction] + np.exp(scores[p]) ###
        #print(model_df.loc[model_df['interval'] == interval, 'score'])
        '''
        for p, prediction in enumerate(predictions):
            interval_results[prediction] = interval_results[prediction] + 1
            if prediction == predictions[-1]:
                interval_results[prediction] = interval_results[prediction] + 0.5
        '''
        #print(interval_results)
        interval_prediction = np.argmax(interval_results)
        intervals_labels.append(interval_label)
        intervals_predictions.append(interval_prediction)
        #print(interval_prediction)

    results = {'label': intervals_labels, 'prediction': intervals_predictions}
    interval_df = pd.DataFrame(results)

    n_samples = len(interval_df)
    intervals_correct = sum(interval_df['label'] == interval_df['prediction'])
    intervals_accuracy = intervals_correct / n_samples * 100

    confusion_matrix = compute_confusion_matrix(d, interval_df)
    performance = compute_performance(confusion_matrix)

    return intervals_accuracy, performance


def compute_accuracy_start_interval(model_df):
    # tot accuracy intervals
    intervals = np.sort(model_df['interval'].unique())
    intervals_labels = []
    intervals_predictions = []

    for i, interval in enumerate(intervals):
        interval_label = model_df.loc[model_df['interval'] == interval, 'label'].iloc[0]
        interval_prediction = model_df.loc[model_df['interval'] == interval, 'prediction'].iloc[0].tolist()

        intervals_labels.append(interval_label)
        intervals_predictions.append(interval_prediction)

    results = {'label': intervals_labels, 'prediction': intervals_predictions}
    interval_df = pd.DataFrame(results)

    n_samples = len(interval_df)
    intervals_correct = sum(interval_df['label'] == interval_df['prediction'])
    intervals_accuracy = intervals_correct / n_samples * 100

    confusion_matrix = compute_confusion_matrix(d, interval_df)
    performance = compute_performance(confusion_matrix)

    return intervals_accuracy, performance


def compute_accuracy_gradual_duration(model_df):
    intervals = np.sort(model_df['interval'].unique())
    intervals_labels_1 = []
    intervals_predictions_1 = []
    intervals_labels_2 = []
    intervals_predictions_2 = []
    intervals_labels_3 = []
    intervals_predictions_3 = []

    for i, interval in enumerate(intervals):
        interval_results = np.zeros(len(d['labels']))
        interval_label = model_df.loc[model_df['interval'] == interval, 'label'].iloc[0]
        predictions = model_df.loc[model_df['interval'] == interval, 'prediction'].tolist()
        scores = model_df.loc[model_df['interval'] == interval, 'score'].tolist()

        for p, prediction in enumerate(predictions):
            interval_results[prediction] = interval_results[prediction] + np.exp(scores[p])

        '''
        for p, prediction in enumerate(predictions):
            interval_results[prediction] = interval_results[prediction]+ 1
            if prediction == predictions[-1]:
                interval_results[prediction] = interval_results[prediction]+ 0.5
        '''
        interval_prediction = np.argmax(interval_results)

        if len(predictions) == 1:
            intervals_labels_1.append(interval_label)
            intervals_predictions_1.append(interval_prediction)
        elif len(predictions) == 2:
            intervals_labels_2.append(interval_label)
            intervals_predictions_2.append(interval_prediction)
        elif len(predictions) >= 3:
            intervals_labels_3.append(interval_label)
            intervals_predictions_3.append(interval_prediction)

    # duration 1
    results_1 = {'label': intervals_labels_1, 'prediction': intervals_predictions_1}
    interval_df_1 = pd.DataFrame(results_1)
    n_samples_1 = len(interval_df_1)
    intervals_correct_1 = sum(interval_df_1['label'] == interval_df_1['prediction'])
    intervals_accuracy_1 = intervals_correct_1 / n_samples_1 * 100
    confusion_matrix_1 = compute_confusion_matrix(d, interval_df_1)
    performance_1 = compute_performance(confusion_matrix_1)

    # duration 2
    results_2 = {'label': intervals_labels_2, 'prediction': intervals_predictions_2}
    interval_df_2 = pd.DataFrame(results_2)
    n_samples_2 = len(interval_df_2)
    intervals_correct_2 = sum(interval_df_2['label'] == interval_df_2['prediction'])
    intervals_accuracy_2 = intervals_correct_2 / n_samples_2 * 100
    confusion_matrix_2 = compute_confusion_matrix(d, interval_df_2)
    performance_2 = compute_performance(confusion_matrix_2)

    # duration 3
    results_3 = {'label': intervals_labels_3, 'prediction': intervals_predictions_3}
    interval_df_3 = pd.DataFrame(results_3)
    n_samples_3 = len(interval_df_3)
    intervals_correct_3 = sum(interval_df_3['label'] == interval_df_3['prediction'])
    intervals_accuracy_3 = intervals_correct_3 / n_samples_3 * 100
    confusion_matrix_3 = compute_confusion_matrix(d, interval_df_3)
    performance_3 = compute_performance(confusion_matrix_3)

    gradual_intervals_accuracies_list = [round(intervals_accuracy_1, 2), round(intervals_accuracy_2, 2),
                                         round(intervals_accuracy_3, 2)]
    gradual_intervals_performances_list = [performance_1, performance_2, performance_3]

    return gradual_intervals_accuracies_list, gradual_intervals_performances_list


def compute_accuracy_incremental_duration(model_df):
    intervals = np.sort(model_df['interval'].unique())
    intervals_labels_1 = []
    intervals_predictions_1 = []
    intervals_labels_2 = []
    intervals_predictions_2 = []
    intervals_labels_3 = []
    intervals_predictions_3 = []
    intervals_labels_4 = []
    intervals_predictions_4 = []

    for i, interval in enumerate(intervals):
        interval_results_1 = np.zeros(len(d['labels']))
        interval_results_2 = np.zeros(len(d['labels']))
        interval_results_3 = np.zeros(len(d['labels']))
        interval_results_4 = np.zeros(len(d['labels']))
        interval_label = model_df.loc[model_df['interval'] == interval, 'label'].iloc[0]
        predictions = model_df.loc[model_df['interval'] == interval, 'prediction'].tolist()
        scores = model_df.loc[model_df['interval'] == interval, 'score'].tolist()

        if len(predictions) >= 3:
            incremental_predictions_4 = predictions
            for p_4, prediction_4 in enumerate(incremental_predictions_4):
                interval_results_4[prediction_4] = interval_results_4[prediction_4] + np.exp(scores[p_4])
            '''
            for p_4, prediction_4 in enumerate(incremental_predictions_4):
                interval_results_4[prediction_4] = interval_results_4[prediction_4] + 1
                if prediction_4 == incremental_predictions_4[-1]:
                    interval_results_4[prediction_4] = interval_results_4[prediction_4] + 0.5
            '''
            interval_prediction_4 = np.argmax(interval_results_4)
            intervals_labels_4.append(interval_label)
            intervals_predictions_4.append(interval_prediction_4)

        if len(predictions) >= 3:
            incremental_predictions_3 = predictions[0:3]
            for p_3, prediction_3 in enumerate(incremental_predictions_3):
                interval_results_3[prediction_3] = interval_results_3[prediction_3] +  + np.exp(scores[p_3])
            '''
            for p_3, prediction_3 in enumerate(incremental_predictions_3):
                interval_results_3[prediction_3] = interval_results_3[prediction_3] + 1
                if prediction_3 == incremental_predictions_3[-1]:
                    interval_results_3[prediction_3] = interval_results_3[prediction_3] + 0.5
            '''
            interval_prediction_3 = np.argmax(interval_results_3)
            intervals_labels_3.append(interval_label)
            intervals_predictions_3.append(interval_prediction_3)

        if len(predictions) >= 2:

            incremental_predictions_2 = predictions[0:2]
            for p_2, prediction_2 in enumerate(incremental_predictions_2):
                interval_results_2[prediction_2] = interval_results_2[prediction_2] +  + np.exp(scores[p_2])
            '''
            for p_2, prediction_2 in enumerate(incremental_predictions_2):
                interval_results_2[prediction_2] = interval_results_2[prediction_2] + 1
                if prediction_2 == incremental_predictions_2[-1]:
                    interval_results_2[prediction_2] = interval_results_2[prediction_2] + 0.5
            '''
            interval_prediction_2 = np.argmax(interval_results_2)
            intervals_labels_2.append(interval_label)
            intervals_predictions_2.append(interval_prediction_2)

        if len(predictions) >= 1:

            #incremental_predictions_1 = [predictions[0]]
            #for p_1, prediction_1 in enumerate(incremental_predictions_1):
            #    interval_results_1[prediction_1] = interval_results_1[prediction_1] + scores[p_1]
            #interval_prediction_1 = np.argmax(interval_results_1)
            interval_prediction_1 = predictions[0]
            intervals_labels_1.append(interval_label)
            intervals_predictions_1.append(interval_prediction_1)

    # duration 1
    results_1 = {'label': intervals_labels_1, 'prediction': intervals_predictions_1}
    interval_df_1 = pd.DataFrame(results_1)
    n_samples_1 = len(interval_df_1)
    intervals_correct_1 = sum(interval_df_1['label'] == interval_df_1['prediction'])
    intervals_accuracy_1 = intervals_correct_1 / n_samples_1 * 100
    confusion_matrix_1 = compute_confusion_matrix(d, interval_df_1)
    performance_1 = compute_performance(confusion_matrix_1)

    # duration 2
    results_2 = {'label': intervals_labels_2, 'prediction': intervals_predictions_2}
    interval_df_2 = pd.DataFrame(results_2)
    n_samples_2 = len(interval_df_2)
    intervals_correct_2 = sum(interval_df_2['label'] == interval_df_2['prediction'])
    intervals_accuracy_2 = intervals_correct_2 / n_samples_2 * 100
    confusion_matrix_2 = compute_confusion_matrix(d, interval_df_2)
    performance_2 = compute_performance(confusion_matrix_2)

    # duration 3
    results_3 = {'label': intervals_labels_3, 'prediction': intervals_predictions_3}
    interval_df_3 = pd.DataFrame(results_3)
    n_samples_3 = len(interval_df_3)
    intervals_correct_3 = sum(interval_df_3['label'] == interval_df_3['prediction'])
    intervals_accuracy_3 = intervals_correct_3 / n_samples_3 * 100
    confusion_matrix_3 = compute_confusion_matrix(d, interval_df_3)
    performance_3 = compute_performance(confusion_matrix_3)

    # duration over 3
    results_4 = {'label': intervals_labels_4, 'prediction': intervals_predictions_4}
    interval_df_4 = pd.DataFrame(results_4)
    n_samples_4 = len(interval_df_4)
    intervals_correct_4 = sum(interval_df_4['label'] == interval_df_4['prediction'])
    intervals_accuracy_4 = intervals_correct_4 / n_samples_4 * 100
    confusion_matrix_4 = compute_confusion_matrix(d, interval_df_4)
    performance_4 = compute_performance(confusion_matrix_4)

    incremental_intervals_accuracies_list = [intervals_accuracy_1, intervals_accuracy_2,
                                             intervals_accuracy_3, intervals_accuracy_4]
    incremental_intervals_performances_list = [performance_1, performance_2, performance_3, performance_4]

    return incremental_intervals_accuracies_list, incremental_intervals_performances_list


def compute_weighted_average(performances, metrics):

    weighted_elements = np.zeros(len(d['labels']))
    n_elements = np.zeros(len(d['labels']))
    all_performance = np.zeros(len(d['labels']))
    for l, label in enumerate(d['labels']):
        performance = performances[metrics].loc[label].item()
        n = performances['n_actual_samples'].loc[label].item()
        all_performance[l] = performance
        n_elements[l] = n
        weighted_elements[l] = n * performance
    all_samples = performances['n_actual_samples'].sum(axis=0).item()
    weighted_average = np.sum(weighted_elements) / all_samples

    return weighted_average


def compute_weighted_F1_average(performances, f1_array):

    weighted_elements = np.zeros(len(d['labels']))
    n_elements = np.zeros(len(d['labels']))
    all_performance = np.zeros(len(d['labels']))
    for l, label in enumerate(d['labels']):
        n = performances['n_actual_samples'].loc[label].item()
        n_elements[l] = n
        all_performance[l] = f1_array[l]
        weighted_elements[l] = n * f1_array[l]
    all_samples = performances['n_actual_samples'].sum(axis=0).item()
    weighted_average = np.sum(weighted_elements) / all_samples

    return weighted_average


def fill_dataframes(m, sequences_accuracy, sequences_performance, tot_intervals_accuracy, tot_intervals_performance,
                    start_intervals_accuracy, start_intervals_performance, gradual_intervals_accuracies_list,
                    gradual_intervals_performances_list, incremental_intervals_accuracies_list,
                    incremental_intervals_performances_list):
    print(sequences_accuracy)
    sequences_results.loc[m, 'accuracy'] = sequences_accuracy
    tot_intervals_results.loc[m, 'accuracy'] = tot_intervals_accuracy
    start_intervals_results.loc[m, 'accuracy'] = start_intervals_accuracy
    print(sequences_results.loc[m, 'accuracy'])
    #print(sequences_accuracy, tot_intervals_accuracy, start_intervals_accuracy)


    gradual_intervals_results.loc[m, 'accuracy_1'] = gradual_intervals_accuracies_list[0]
    gradual_intervals_results.loc[m, 'accuracy_2'] = gradual_intervals_accuracies_list[1]
    gradual_intervals_results.loc[m, 'accuracy_3'] = gradual_intervals_accuracies_list[2]

    incremental_intervals_results.loc[m, 'accuracy_1'] = incremental_intervals_accuracies_list[0]
    incremental_intervals_results.loc[m, 'accuracy_2'] = incremental_intervals_accuracies_list[1]
    incremental_intervals_results.loc[m, 'accuracy_3'] = incremental_intervals_accuracies_list[2]
    incremental_intervals_results.loc[m, 'accuracy_4'] = incremental_intervals_accuracies_list[3]

    for l, label in enumerate(d['labels']):

        sequences_results.loc[m, '{}_recall'.format(label)] = sequences_performance['recall'].loc[label].item()
        sequences_results.loc[m, '{}_precision'.format(label)] = sequences_performance['precision'].loc[label].item()

        tot_intervals_results.loc[m, '{}_recall'.format(label)] = tot_intervals_performance['recall'].loc[label].item()
        tot_intervals_results.loc[m, '{}_precision'.format(label)] = tot_intervals_performance['precision'].loc[label].item()

        start_intervals_results.loc[m, '{}_recall'.format(label)] = start_intervals_performance['recall'].loc[label].item()
        start_intervals_results.loc[m, '{}_precision'.format(label)] = start_intervals_performance['precision'].loc[label].item()

        incremental_intervals_results.loc[m, '{}_recall_1'.format(label)] = incremental_intervals_performances_list[0]['recall'].loc[label].item()
        incremental_intervals_results.loc[m, '{}_precision_1'.format(label)] = incremental_intervals_performances_list[0]['precision'].loc[label].item()

        incremental_intervals_results.loc[m, '{}_recall_2'.format(label)] = incremental_intervals_performances_list[1]['recall'].loc[label].item()
        incremental_intervals_results.loc[m, '{}_precision_2'.format(label)] = incremental_intervals_performances_list[1]['precision'].loc[label].item()

        incremental_intervals_results.loc[m, '{}_recall_3'.format(label)] = incremental_intervals_performances_list[2]['recall'].loc[label].item()
        incremental_intervals_results.loc[m, '{}_precision_3'.format(label)] = incremental_intervals_performances_list[2]['precision'].loc[label].item()

        incremental_intervals_results.loc[m, '{}_recall_4'.format(label)] = incremental_intervals_performances_list[3]['recall'].loc[label].item()
        incremental_intervals_results.loc[m, '{}_precision_4'.format(label)] = incremental_intervals_performances_list[3]['precision'].loc[label].item()

    sequences_results.loc[m, 'weighted_recall'] = compute_weighted_average(sequences_performance, 'recall')
    sequences_results.loc[m, 'weighted_precision'] = compute_weighted_average(sequences_performance, 'precision')

    tot_intervals_results.loc[m, 'weighted_recall'] = compute_weighted_average(tot_intervals_performance, 'recall')
    tot_intervals_results.loc[m, 'weighted_precision'] = compute_weighted_average(tot_intervals_performance, 'precision')

    start_intervals_results.loc[m, 'weighted_recall'] = compute_weighted_average(start_intervals_performance, 'recall')
    start_intervals_results.loc[m, 'weighted_precision'] = compute_weighted_average(start_intervals_performance, 'precision')

    incremental_intervals_results.loc[m, 'weighted_recall_1'] = compute_weighted_average(incremental_intervals_performances_list[0],
                                                                                         'recall')
    incremental_intervals_results.loc[m, 'weighted_precision_1'] = compute_weighted_average(incremental_intervals_performances_list[0],
                                                                                                          'precision')

    incremental_intervals_results.loc[m, 'weighted_recall_2'] = compute_weighted_average(incremental_intervals_performances_list[1],
                                                                                         'recall')
    incremental_intervals_results.loc[m, 'weighted_precision_2'] = compute_weighted_average(incremental_intervals_performances_list[1],
                                                                                                          'precision')

    incremental_intervals_results.loc[m, 'weighted_recall_3'] = compute_weighted_average(incremental_intervals_performances_list[2],
                                                                                         'recall')
    incremental_intervals_results.loc[m, 'weighted_precision_3'] = compute_weighted_average(incremental_intervals_performances_list[2],
                                                                                                          'precision')

    incremental_intervals_results.loc[m, 'weighted_recall_4'] = compute_weighted_average(incremental_intervals_performances_list[3],
                                                                                         'recall')
    incremental_intervals_results.loc[m, 'weighted_precision_4'] = compute_weighted_average(incremental_intervals_performances_list[3],
                                                                                                          'precision')

    compute_f1(m, sequences_performance, tot_intervals_performance, start_intervals_performance, incremental_intervals_performances_list)

    return


def fill_icub_dataframe(sequences_accuracy, sequences_performance):

    global icub_ds
    empty = np.zeros(0)
    icub_ds = pd.Series([])
    icub_ds['accuracy'] = sequences_accuracy
    f1_array = np.zeros(len(d['labels']))

    for l, label in enumerate(d['labels']):

        icub_ds['{}_recall'.format(label)] = sequences_performance['recall'].loc[label].item()
        icub_ds['{}_precision'.format(label)] = sequences_performance['precision'].loc[label].item()
        f1score = f1_formula(sequences_performance['recall'].loc[label].item(),
                                                         sequences_performance['precision'].loc[label].item())
        icub_ds['{}_f1score'.format(label)] = f1score
        f1_array[l] = f1score


    icub_ds['weighted_avg_f1score'] = compute_weighted_F1_average(sequences_performance, f1_array)
    icub_ds['weighted_recall'] = compute_weighted_average(sequences_performance, 'recall')
    icub_ds['weighted_precision'] = compute_weighted_average(sequences_performance, 'recall')

    return


def plot_results():

    print(d['version'])
    if d['version'] == 'vTHESIS_50' or d['version'] == 'vTHESIS_50_512b':
        bar_plot_classes(d, tot_intervals_results, d['version'], d['model'])

    if d['version'] == 'vTHESIS_50_binary':
        bar_plot_binary(d, sequences_results, d['version'], d['model'])

    bar_IJCNN(d)
    trend_incremental_accuracy(d, d['model'])

    return


def plot_icub_results(sequences_performance):

    plot_confusion_matrix_(d, sequences_performance, icub_ds)

    return


def fill_dataframes_comparison(m, tot_intervals_accuracy, tot_intervals_performance):

    f1_array = np.zeros(len(d['labels']))
    for l, label in enumerate(d['labels']):
        f1_array[l] = f1_formula(tot_intervals_performance['recall'].loc[label].item(),
                                 tot_intervals_performance['precision'].loc[label].item())
    if d['version'] == 'vTHESIS_50_binary':
        binary_df.loc[m, 'accuracy_{}'.format(d['model'])] = tot_intervals_accuracy
        binary_df.loc[m, 'weighted_avg_f1score_{}'.format(d['model'])] = compute_weighted_F1_average(tot_intervals_performance,
                                                                                        f1_array)

    elif d['version'] == 'vTHESIS_50_512b':

        three_classes_df.loc[m, 'accuracy_{}'.format(d['model'])] = tot_intervals_accuracy
        three_classes_df.loc[m, 'weighted_avg_f1score_{}'.format(d['model'])] = compute_weighted_F1_average(tot_intervals_performance,
                                                                                               f1_array)

    return



def main(args):
    read_arguments(args)
    read_variables()

    create_dataframes()
    for m, model in enumerate(d['models_list']):
        d['model_name'] = model
        model_df = pd.read_csv(os.path.join(d['results_dir'], d['model_name'], d['results_file']), sep='\t',
                               index_col='index')
        model_df.rename(columns={'predictions': 'prediction'}, inplace=True)
        model_df = adjust_intervals(model_df)

        sequences_accuracy, sequences_performance = compute_accuracy_sequence(model_df)
        tot_intervals_accuracy, tot_intervals_performance = compute_accuracy_tot_interval(model_df)
        start_intervals_accuracy, start_intervals_performance = compute_accuracy_start_interval(model_df)
        gradual_intervals_accuracies_list, gradual_intervals_performances_list = compute_accuracy_gradual_duration(
            model_df)
        incremental_intervals_accuracies_list, incremental_intervals_performances_list = compute_accuracy_incremental_duration(
            model_df)

        fill_dataframes(m, sequences_accuracy, sequences_performance, tot_intervals_accuracy,
                        tot_intervals_performance,
                        start_intervals_accuracy, start_intervals_performance, gradual_intervals_accuracies_list,
                        gradual_intervals_performances_list, incremental_intervals_accuracies_list,
                        incremental_intervals_performances_list)

    plot_results()

    print("computed metrics on sequences ", sequences_results.columns)
    print("computed metrics on incremental intervals ", incremental_intervals_results.columns)

    print('')
    print("weighted_avg_f1score mean sequences", np.average(sequences_results['weighted_avg_f1score']))
    print("weighted_avg_f1score std sequences", np.std(sequences_results['weighted_avg_f1score']))
    print('')
    print("weighted_avg_f1score mean utterances", np.average(tot_intervals_results['weighted_avg_f1score']))
    print("weighted_avg_f1score std utterances", np.std(tot_intervals_results['weighted_avg_f1score']))
    print('')
    print("weighted_avg_f1score mean first sequence", np.average(start_intervals_results['weighted_avg_f1score']))
    print("weighted_avg_f1score std first sequence", np.std(start_intervals_results['weighted_avg_f1score']))
    print('')

    print("ROBOT recall sequences", np.average(sequences_results['ROBOT_recall']))
    print("ROBOT precision sequences", np.average(sequences_results['ROBOT_precision']))
    print("ROBOT F1-score sequences", np.average(sequences_results['ROBOT_f1score']))

    print('')
    print("LEFT recall sequences", np.average(tot_intervals_results['LEFT_recall']))
    print("LEFT precision sequences", np.average(tot_intervals_results['LEFT_precision']))
    print("LEFT F1-score sequences", np.average(sequences_results['LEFT_f1score']))

    print('')
    print("RIGHT recall sequences", np.average(start_intervals_results['RIGHT_recall']))
    print("RIGHT precision sequences", np.average(sequences_results['RIGHT_precision']))
    print("RIGHT F1-score sequences", np.average(sequences_results['RIGHT_f1score']))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, default='/usr/local/src/robot/cognitiveinteraction/addressee_estimation_ijcnn23/results')
    parser.add_argument('--model', type=str, default='cnn_lstm_pose')
    parser.add_argument('--version', type=str, default='vTHESIS_50')
    parser.add_argument('--results_file', type=str, default='results.csv')
    parser.add_argument('--sequence_duration', type=float, default=0.8)
    parser.add_argument('--save_plot_dir', type=str, default='/home/icub/Documents/Carlo/THESIS/PLOTS')

    args = parser.parse_args()

    main(args)
