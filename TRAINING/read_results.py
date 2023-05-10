from test import test_solo
import pandas as pd
import numpy as np
import os
import argparse
from utils import calculate_performance
from plots import plot_confusion_matrix, plot_length_interval_impact


def read_args(args):

    global a
    a = {
        'results_dir': args.results_dir,
        'class_names': args.class_names,
        'model_dir': args.model_dir,
        'model': args.model,
        'version': args.version,
        'model_file': args.model_file
    }

    return


def set_variables1():

    global all_lengths_df, models_path
    all_lengths_df = pd.DataFrame(columns=['length_interval', 'n_data', 'correct'])
    models_path = os.path.join(a['model_dir'], a['model'], a['version'])

    return


def set_variables2(model):

    a['model_name'] = model
    result_path = os.path.join(a['results_dir'], a['model_name'])
    a['plot_dir'] = os.path.join(result_path, 'plot')
    if not os.path.exists(a['plot_dir']):
        os.mkdir(a['plot_dir'])

    return result_path


def read_model_results(result_path):

    results = np.load(os.path.join(result_path, a['model_file']), allow_pickle=True)
    #print(results, '\n\n\n')
    results_df = pd.read_csv(os.path.join(result_path, 'results.csv'), sep="\t", index_col='index')

    return results_df


def divide_for_intervals(results_df):

    all_intervals_df = pd.DataFrame(columns=results_df.columns)

    for i, interval in enumerate(results_df['interval']):
        if i > 0 and interval < results_df['interval'].loc[i-1]:
            addition = results_df['interval'].loc[i-1]
            results_df['interval'].loc[i:] = results_df['interval'].loc[i:]+addition
            break
    intervals = results_df['interval'].unique()

    all_labels = []
    all_predictions = []
    all_scores = []
    all_n_sequences = []
    all_reliabilities = []

    for i, interval in enumerate(intervals):

        results_interval = np.zeros(3)
        label = results_df[results_df['interval'] == interval]['label'].unique()
        preds = results_df[results_df['interval'] == interval]['predictions'].tolist()
        scores = results_df[results_df['interval'] == interval]['score'].tolist()

        for p, pred in enumerate(preds):

            results_interval[pred] = results_interval[pred] + scores[p]
            #results_interval[pred] = results_interval[pred] + 1
            #if preds[p] != preds[0]:
            #    results_interval[pred] = results_interval[pred] + scores[p]
            #else:
            #    results_interval[pred] = results_interval[pred] + scores[p]*1.5

        prediction = np.argmax(results_interval)
        score = np.amax(results_interval)
        n_sequence = len(preds)
        reliability = score / np.sum(results_interval) * 100
        #print(label, prediction, score, n_sequence, interval, reliability)

        all_labels.append(label[0])
        all_predictions.append(prediction)
        all_scores.append(score)
        all_n_sequences.append(n_sequence)
        all_reliabilities.append(reliability)

    res = {'label': all_labels, 'prediction': all_predictions, 'score': all_scores, 'n_sequences': all_n_sequences, 'interval': intervals, 'reliability': all_reliabilities}
    interval_df = pd.DataFrame(res)

    return interval_df


def calculate_confusion_matrix(a, interval_df):

    confusion_matrix = np.zeros((len(a['class_names']), len(a['class_names'])))
    total = len(interval_df)
    correct = sum(interval_df['label'] == interval_df['prediction'])
    print('accuracy {} %'.format(correct / total * 100))
    for i, l in enumerate(interval_df['label']):
        confusion_matrix[int(l), int(interval_df['prediction'].loc[i])] += 1
    return confusion_matrix


def calculate_confusion_matrix_cleaned(a, interval_df):

    cleaned_interval_df = interval_df
    #print(cleaned_interval_df)
    confusion_matrix_cleaned = np.zeros((len(a['class_names']), len(a['class_names'])))
    total = len(cleaned_interval_df)
    correct = sum(cleaned_interval_df['label'] == cleaned_interval_df['prediction'])
    print('cleaned accuracy {} %'.format(correct / total * 100))
    for i, l in enumerate(cleaned_interval_df['label']):
        print(i, l)
        confusion_matrix_cleaned[int(l), int(cleaned_interval_df['prediction'].loc[i])] += 1

    return confusion_matrix_cleaned


def length_utterance_df_creation(interval_df):

    global all_lengths_df
    n_sequences = np.sort(interval_df['n_sequences'].unique())
    for n, seq in enumerate(n_sequences):
        df = interval_df[interval_df['n_sequences']==seq]

        total = len(df)
        correct = sum(df['label'] == df['prediction'])
        accuracy = round(correct/total*100, 2)
        data_length_interval = {'length_interval': [seq], 'n_data': [total], 'correct': [correct]}
        length_df = pd.DataFrame(data_length_interval)
        all_lengths_df = pd.concat([all_lengths_df, length_df], ignore_index=True)

    return


def length_utterance_impact():

    length_interval = np.sort(all_lengths_df['length_interval'].unique())
    #print(all_lengths_df[all_lengths_df['length_interval'] == 10])

    all_totals = []
    all_corrects = []
    all_accuracies = []

    for n, length in enumerate(length_interval):
        total = sum(all_lengths_df[all_lengths_df['length_interval'] == length]['n_data'])
        correct = sum(all_lengths_df[all_lengths_df['length_interval'] == length]['correct'])
        accuracy = round((correct/total * 100), 2)
        all_totals.append(total)
        all_corrects.append(correct)
        all_accuracies.append(accuracy)

        #print(length, accuracy)
    plot_length_interval_impact(a, length_interval, all_totals, all_accuracies)

    return


def main(args):

    read_args(args)

    set_variables1()
    models_list = [d for d in os.listdir(a['results_dir']) if os.path.isdir(os.path.join(a['results_dir'], d))]
    print(models_list)
    models_list.sort()
    for model in models_list:

        result_path = set_variables2(model)
        results_df = read_model_results(result_path)

        interval_df = divide_for_intervals(results_df)
        interval_df.to_csv(os.path.join(result_path, 'interval_results.csv'), sep='\t', index_label='index')

        a['confusion_matrix'] = calculate_confusion_matrix(a, interval_df)
        #a['confusion_matrix_cleaned'] = calculate_confusion_matrix_cleaned(a, interval_df)
        length_utterance_df_creation(interval_df)
    length_utterance_impact()

    performance_test = calculate_performance(a)
    plot_confusion_matrix(a, performance_test)




parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, default='/home/icub/Documents/Carlo/results/cnn_lstm_mm_v13_reproducible')
parser.add_argument('--model_dir', type=str, default='/home/icub/Documents/Carlo/models/FINALS')
parser.add_argument('--model', type=str, default='cnn_lstm_mm') #cnn_lstm_image #cnn_audio
parser.add_argument('--version', type=str, default='v_21_nao_left_right_softmaxrelu')
parser.add_argument('--model_file', type=str, default='cnn_lstm_mm_info.npy')
parser.add_argument('--class_names', type=list, default=['NAO', 'PLEFT', 'PRIGHT'])
args = parser.parse_args()

main(args)
