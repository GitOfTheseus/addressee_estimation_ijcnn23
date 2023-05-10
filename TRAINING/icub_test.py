from test import test_solo
import pandas as pd
import numpy as np
import os
import argparse


def read_args(args):
    a = {
        'model_path': args.model_path,
        'model': args.model,
        'version': args.version,
        'only_one': args.only_one,
        'model_name': args.model_name,
        'model_file': args.model_file,
        'test_filename': args.test_filename,
        'labels_dir': args.labels_dir,
        'data_dir': args.data_dir,
        'save_results_dir': args.save_results_dir
    }

    return a


def save_results(a, d, error_df, intervals_check_df):

    save_dir = os.path.join(a['save_results_dir'], a['model'], a['version'])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, a['model_name'])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    np.save(os.path.join(save_dir, '{}_info.npy'.format(a['model'])), d)
    print('Model INFO updated in {}_info.npy'.format(a['model']))
    error_df.to_csv(os.path.join(save_dir, 'errors.csv'), sep='\t', index_label='index')
    intervals_check_df.to_csv(os.path.join(save_dir, 'results.csv'), sep='\t', index_label='index')


def main(args):

    a = read_args(args)
    a['slot_list'] = ['09', '10', '12', '15', '18', '19', '24', '26', '27', '30']
    a['model_dir'] = os.path.join(a['model_path'], a['model'], a['version'], a['model_name'], a['model_file'])

    d, error_df, intervals_check_df = test_solo(dict_dir=a['model_dir'], test_filename=a['test_filename'],
                                                data_dir=a['data_dir'], labels_dir=a['labels_dir'])
    save_results(a, d, error_df, intervals_check_df)


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='/home/icub/Documents/Carlo/models/FINALS')
parser.add_argument('--model', type=str, default='cnn_lstm_image') #cnn_lstm_image #cnn_audio
parser.add_argument('--version', type=str, default='v19_complete_training_nao_left_right')
parser.add_argument('--only_one', type=bool, default=True)
parser.add_argument('--model_name', type=str, default='cnn_lstm_image_training_complete2022-11-25_20-27')
parser.add_argument('--model_file', type=str, default='lstm_image_info.npy')
parser.add_argument('--test_filename', type=str, default='test_icub_3_mean.csv')  #test_icub_1.csv
parser.add_argument('--labels_dir', type=str, default='/home/icub/Documents/Carlo/icub_dataset/dataset')
parser.add_argument('--data_dir', type=str, default='/home/icub/Documents/Carlo/icub_dataset/dataset')
parser.add_argument('--save_results_dir', type=str, default='/home/icub/Documents/Carlo/results')
args = parser.parse_args()

main(args)
