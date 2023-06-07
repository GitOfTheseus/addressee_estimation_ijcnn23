import numpy as np
import os
import argparse
from B_TRAIN.test import test_solo

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

    save_dir = os.path.join(a['save_results_dir'], a['model'])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, a['version'])
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
    a['model_path'] = os.path.join(a['model_path'], a['model'], a['version'])
    #a['labels_dir'] = os.path.join(a['labels_dir'], 'csv_{}'.format(a['version']))
    models_list = [d for d in os.listdir(a['model_path']) if os.path.isdir(os.path.join(a['model_path'], d))]
    models_list.sort()
    print(a['model_path'])

    for n, model in enumerate(models_list):
        a['model_name'] = model
        a['model_dir'] = os.path.join(a['model_path'], model, a['model_file'])
        print(a['model_dir'])
        a['test_filename'] = 'test_slot_{}_out.csv'.format(a['slot_list'][n])
        print(a['test_filename'])

        #a['data_dir'] = os.path.join(a['data_dir'], a['slot_list'][n])

        d, error_df, intervals_check_df = test_solo(dict_dir=a['model_dir'], test_filename=a['test_filename'],
                                                    data_dir=a['data_dir'], labels_dir= a['labels_dir'])
        save_results(a, d, error_df, intervals_check_df)


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='/usr/local/src/robot/cognitiveinteraction/addressee_estimation_ijcnn23/models')
parser.add_argument('--model', type=str, default='cnn_lstm_pose') #cnn_lstm_image #cnn_audio
parser.add_argument('--version', type=str, default='vTHESIS_50')
parser.add_argument('--only_one', type=bool, default=False)
parser.add_argument('--model_name', type=str, default='cnn_lstm_mm_9out_2022-11-20_16-29')
parser.add_argument('--model_file', type=str, default='lstm_pose_info.npy')
parser.add_argument('--test_filename', type=str, default='')  #test_icub_1.csv
parser.add_argument('--labels_dir', type=str, default='/usr/local/src/robot/cognitiveinteraction/addressee_estimation_ijcnn23/labels')
parser.add_argument('--data_dir', type=str, default='/home/icub/Documents/Carlo/dataset_slots')
parser.add_argument('--save_results_dir', type=str, default='/usr/local/src/robot/cognitiveinteraction/addressee_estimation_ijcnn23/results')
args = parser.parse_args()

main(args)
