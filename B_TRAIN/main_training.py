import numpy as np
import argparse
import torch
import random
from B_TRAIN.loaders import data_loader_training
from B_TRAIN.utils import read_arguments, create_paths, name_csv_files, save
from B_TRAIN.train import train
from B_TRAIN.test import test

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)

def main(args):

    d = read_arguments(args)

    if not d['training_complete']:  # if the training is not complete for all the slots but you are following the 10-fold cross validation approach

        if d['all_slots_training']: # if you want to train 10 models one after the other, leaving out every time a different slot for the testing
            slot_test_list = range(10)
        else:
            slot_test_list = [d['slot_test']] # if you want to train only one model on nine slots, leaving out one specific slot (d['slot_test']

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
    parser.add_argument('--model', type=str, default='cnn_lstm_mm')
    #options:
    # cnn_lstm_mm: intermediate fusion face+pose;
    # cnn_lstm_mm_latefusion: late fusion face+pose;
    # cnn_lstm_image: only face;
    # cnn_lstm_pose: only pose;

    parser.add_argument('--label_filename', type=str, default='lstm_label_3_classes.csv')
    # the label filename resulting from the A_FEATURES_EXTRACTION codes.
    # But if you already have the final label files for the 10 folds cross validation you don't need this: go in loaders.data_loader_training and comment
    # join_slot
    # shuffle_dataset
    # create_eval_group

    parser.add_argument('--data_dir', type=str, default='[your directory]/dataset_slots')
    #options
    # 3classes 'lstm_label_no_doubles_augmented_nogroup_3.csv'
    # binary 'lstm_label_no_doubles_involved.csv'

    parser.add_argument('--models_dir', type=str, default='[your directory]/models')
    parser.add_argument('--class_names', type=list, default=['NAO', 'PLEFT', 'PRIGHT'])
    # options
    # 3classes ['NAO', 'PLEFT', 'PRIGHT']
    # binary ['INVOLVED', 'NOT_INVOLVED']

    parser.add_argument('--solo_audio', type=bool, default=False)
    parser.add_argument('--all_slots_training', type=bool, default=True)
    parser.add_argument('--slot_test', type=int, default=0)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--n_seq', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--layer_dim', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--step_size', type=int, default=40) 
    parser.add_argument('--optimizer1', type=str, default='SGD_opt')
    parser.add_argument('--optimizer2', type=str, default='Adam_opt')
    # options
    # SGD_opt 
    # RMSprop_opt 
    # Adam_opt
    
    parser.add_argument('--dict_dir', type=str, default='')
    parser.add_argument('--training_complete', type=bool, default=False)

    args = parser.parse_args()
    main(args)

