from loaders import data_loader_training
import argparse
from utils import read_arguments, create_paths, name_csv_files, save
from train import train
from test import test

import torch
import random
import numpy as np

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)

def main(args):

    d = read_arguments(args)

    if not d['training_complete']:

        if d['all_slots_training']:
            slot_test_list = range(10)
        else:
            slot_test_list = [d['slot_test']]

        for slot_test in slot_test_list:
            d['slot_test'] = slot_test
            d = create_paths(d)
            slot_folders, csv_training, test_filename = name_csv_files(data_dir=d['data_dir'], slot_test=d['slot_test'],
                                                                       training_complete=d['training_complete'])
            dataloader = data_loader_training(d, slot_folders, csv_training, test_filename)
            performances, models_list = train(d, dataloader)
            dict_dir = save(d, performances, models_list)
            print(dict_dir)
            test(dict_dir=d['dict_dir'], test_filename=test_filename)

    else:
        slot_test_list = []
        d['slot_test'] = []
        d = create_paths(d)
        slot_folders, csv_training, test_filename = name_csv_files(data_dir=d['data_dir'], slot_test=d['slot_test'],
                                                                   training_complete=d['training_complete'])
        dataloader = data_loader_training(d, slot_folders, csv_training, test_filename)
        performances, models_list = train(d, dataloader)
        dict_dir = save(d, performances, models_list)
        print(dict_dir)
        print('TRAINING COMPLETE')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn_lstm_pose')  #cnn_lstm_image #cnn_audio
    parser.add_argument('--data_dir', type=str, default='/home/icub/Documents/Carlo/dataset_slots')
    parser.add_argument('--label_filename', type=str, default='lstm_label_no_doubles_augmented_nogroup_3.csv') #all_labels_audio_screened_nao  #'lstm_label_no_doubles_augmented_nogroup_3.csv'
    parser.add_argument('--models_dir', type=str, default='/home/icub/Documents/Carlo/models')
    parser.add_argument('--class_names', type=list, default=['NAO', 'LEFT', 'RIGHT']) # ['NAO', 'PLEFT', 'PRIGHT', 'GROUP] # ['INVOLVED', 'NOT_INVOLVED'] # ['SINGLE', 'MANY']
    parser.add_argument('--solo_audio', type=bool, default=True)
    parser.add_argument('--all_slots_training', type=bool, default=False)
    parser.add_argument('--slot_test', type=int, default=0)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--n_seq', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=2) #30
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--layer_dim', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001) #0.001
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--step_size', type=int, default=40)  #40
    parser.add_argument('--optimizer1', type=str, default='SGD_opt') #SGD_opt
    parser.add_argument('--optimizer2', type=str, default='Adam_opt')  # SGD_opt # RMSprop_opt  # Adam_opt
    parser.add_argument('--dict_dir', type=str, default='')
    parser.add_argument('--training_complete', type=bool, default=False)

    args = parser.parse_args()
    main(args)

