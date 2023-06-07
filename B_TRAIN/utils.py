from datetime import datetime
import numpy as np
import pandas as pd
import os
import random
from torchvision import transforms
import torch
import sys

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def read_arguments(args):

    d = {
        'model': args.model,
        'data_dir': args.data_dir,
        'label_filename': args.label_filename,
        'models_dir': args.models_dir,
        'class_names': args.class_names,
        'num_classes': len(args.class_names),
        'solo_audio': args.solo_audio,
        'all_slots_training': args.all_slots_training,
        'training_complete': args.training_complete,
        'slot_test': args.slot_test,
        'seq_len': args.seq_len,
        'n_seq': args.n_seq,
        'num_epochs': args.num_epochs,
        'hidden_dim': args.hidden_dim,
        'layer_dim': args.layer_dim,
        'patience': args.patience,
        'batch': args.n_seq * args.seq_len,
        'learning_rate': args.learning_rate,
        'step_size': args.step_size,
        'optimizer1': args.optimizer1,
        'optimizer2': args.optimizer2,
    }

    return d


def create_paths(d):

    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M")
    if d['training_complete']:
        d['model_name'] = d['model'] + '_training_complete' + date_time_str
    else:
        d['model_name'] = d['model'] + '_' + str(d['slot_test']) + 'out_' + date_time_str

    save_dir = os.path.join(d['models_dir'], d['model'], d['model_name'])
    print(save_dir)
    print(d['model'])
    print(d['model_name'])
    if not os.path.exists(os.path.join(d['models_dir'], d['model'])):
        os.mkdir(os.path.join(d['models_dir'], d['model']))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plot_dir = os.path.join(save_dir, 'plots')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    d['save_dir'] = save_dir
    d['plot_dir'] = plot_dir

    return d


def name_csv_files(data_dir, slot_test, training_complete):

    slot_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    slot_folders.sort()

    if training_complete:
        train_filename = f'train_training_complete.csv'
        eval_filename = f'eval_training_complete.csv'
        test_filename = ''
    else:
        train_filename = f'train_slot_{slot_folders[slot_test]}_out.csv'
        eval_filename = f'eval_slot_{slot_folders[slot_test]}_out.csv'
        test_filename = f'test_slot_{slot_folders[slot_test]}_out.csv'
    csv_training = [train_filename, eval_filename]

    return slot_folders, csv_training, test_filename


def join_slot(model, data_dir, label_filename, slot_test, slot_folders, train_filename, test_filename,
              training_complete):

    print("JOINING SLOT FOR TRAINING AND TEST...")
    #slot_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    #slot_folders.sort()

    #train_csv = f'train_slot_{slot_folders[slot_test]}_out.csv'
    #eval_csv = f'eval_slot_{slot_folders[slot_test]}_out.csv'
    #test_csv = f'test_slot_{slot_folders[slot_test]}_out.csv'
    #csv_training = [train_csv, eval_csv]

    if training_complete:
            df_format = pd.read_csv(os.path.join(data_dir, slot_folders[0], label_filename),
                                sep="\t", index_col='index')
    else:
            df_format = pd.read_csv(os.path.join(data_dir, slot_folders[slot_test], label_filename),
                                sep="\t", index_col='index')

    train_df = pd.DataFrame(columns=df_format.columns)
    if not training_complete:
        test_df = pd.DataFrame(columns=df_format.columns)

    last_sequence = 0
    train_list =[]

    for slot in slot_folders:
        slot_df = pd.read_csv(os.path.join(data_dir, slot, label_filename), sep="\t", index_col='index')
        slot_df['SEQUENCE'] = slot_df['SEQUENCE'] + last_sequence + 1
        last_sequence = slot_df['SEQUENCE'].loc[slot_df.index[-1]]

        if training_complete:
            train_df = pd.concat([train_df, slot_df], ignore_index=True)
            train_list.append(slot)
        else:
            if slot == slot_folders[slot_test]:
                test_df = pd.concat([test_df, slot_df], ignore_index=True)
            else:
                train_df = pd.concat([train_df, slot_df], ignore_index=True)
                train_list.append(slot)

    train_df.to_csv(os.path.join(data_dir, train_filename), sep="\t", index_label='index')
    if not training_complete:
        test_df.to_csv(os.path.join(data_dir, test_filename), sep="\t", index_label='index')

    print("...SLOT JOINT FOR TRAINING AND TEST")

    return


def shuffle_dataset(model, data_dir, label_filename):

    print("SHUFFLING LABELS...")

    train_df = pd.read_csv(os.path.join(data_dir, label_filename), sep="\t", index_col='index')
    shuffled_df = pd.DataFrame(columns=train_df.columns)
    if model == 'cnn_audio':
        n_data = len(train_df.index)
        shuffled_df = train_df.sample(frac=1)

    else:
        n_sequences = train_df['SEQUENCE'].loc[train_df.index[-1]]
        sequence_list = list(range(1, n_sequences + 1))
        random.shuffle(sequence_list)

        for seq in sequence_list:
            shuffled_df = pd.concat([shuffled_df, train_df[train_df['SEQUENCE'] == seq]], ignore_index=True)

    shuffled_df.to_csv(os.path.join(data_dir, label_filename), sep="\t", index_label='index')
    print("...LABELS SHUFFLED")

    return


def create_eval_group(model, data_dir, train_filename, eval_filename, tot_sequences, class_names):

    print("CREATING EVAL...")

    train_df = pd.read_csv(os.path.join(data_dir, train_filename), sep="\t", index_col='index')
    eval_df = pd.DataFrame(columns=train_df.columns)

    n_sequences = train_df['SEQUENCE'].max()
    print(n_sequences)
    sequence_list = list(range(1, n_sequences + 1))
    random.shuffle(sequence_list)
    eval_seq = np.array([])

    for name in class_names:
        sequences = np.unique(train_df[train_df['ADDRESSEE'] == name]['SEQUENCE'].to_numpy())
        random.shuffle(sequences)
        eval_seq = np.concatenate((eval_seq, sequences[0:tot_sequences]))
    random.shuffle(eval_seq)
    for seq in eval_seq:
        eval_df = pd.concat([eval_df, train_df[train_df['SEQUENCE'] == seq]], ignore_index=True)
        train_df = train_df.drop(train_df[train_df['SEQUENCE'] == seq].index)

    eval_df.to_csv(os.path.join(data_dir, eval_filename), sep="\t", index_label='index')
    train_df.to_csv(os.path.join(data_dir, train_filename), sep="\t", index_label='index')

    print("...EVAL CREATED")

    return


def get_transformations():

    data_transform_face = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.GaussianBlur(1, sigma=(0.1, 2.0)),
            transforms.RandomInvert(p=0.1),
            transforms.RandomAdjustSharpness(2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'eval': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    }

    data_transform_pose = {
        'train': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'eval': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
        ])
    }

    return data_transform_face, data_transform_pose


def calculate_possible_shift(df, pose_path, sequence):

    poses = df.loc[df['SEQUENCE']==sequence, 'POSE_SPEAKER']
    minima = []
    maxima = []
    for pose_file in poses:
        pose_dir = os.path.join(pose_path, pose_file)
        pose = np.load(pose_dir)
        to_delete = []
        for k, p in enumerate(pose):
            if p[2] == 0:
                to_delete.append(k)
        pose = np.delete(pose, to_delete, 0)
        minimum = min(pose[:, 0])
        maximum = max(pose[:, 0])
        minima.append(minimum)
        maxima.append(maximum)
    left_max_shift = -1-min(minima)
    right_max_shift = 1-max(maxima)

    return left_max_shift, right_max_shift



def translate_pose(pose, do_shift, left_max_shift, right_max_shift, last_shift):

    #new_pose = pose
    #to_delete = []
    #for k, p in enumerate(pose):
    #    if p[2] == 0:
    #        to_delete.append(k)

    #new_pose = np.delete(new_pose, to_delete, 0)

    #minimum = min(new_pose[:,0])
    #maximum = max(new_pose[:, 0])
    #print(minimum, maximum)
    #left_max_shift = -0.70-minimum
    #right_max_shift = 0.70-maximum
    if do_shift:
        shift = random.uniform(left_max_shift, right_max_shift)
    else:
        shift = last_shift
    last_shift = shift

    #print("shift", shift)
    #print("1",pose)
    for k, p in enumerate(pose):
        if p[2] != 0:
            p[0] = p[0] + shift
            if p[0]<-1 or p[0]>1:
                sys.exit("Error message")
        pose[k, :] = p

    return pose, last_shift

def save(d, performances, models_list):

    model_dir_list = []
    for n, model in enumerate(models_list):
        model['name'] = model['name']
        model_dir = os.path.join(d['save_dir'], model['name'] + str(n) + '.pth')
        model_dir_list.append(model_dir)
        torch.save(model['state'], model_dir)
        print('Model saved as {}/{}.pth'.format(d['model_name'], model['name']))
    d['model_dir_list'] = model_dir_list

    d['performances'] = performances
    d['dict_dir'] = os.path.join(d['save_dir'], '{}_info.npy'.format(model['name']))

    np.save(d['dict_dir'], d)
    print('Model INFO saved in {}_info.npy'.format(model['name']))

    return d


def calculate_performance(d):

    confusion_matrix = d['confusion_matrix']
    total_predicted = confusion_matrix.sum(0)
    total_actual = confusion_matrix.sum(1)
    totals = confusion_matrix.sum(1).sum(0)

    df = pd.DataFrame(confusion_matrix, index=d['class_names'], columns=d['class_names'])
    df_pred = pd.DataFrame(total_predicted, index=d['class_names'])
    df_actual = pd.DataFrame(total_actual, index=d['class_names'])

    total_acc = 0
    recall = pd.DataFrame(np.zeros(len(d['class_names'])), index=d['class_names'])
    precision = pd.DataFrame(np.zeros(len(d['class_names'])), index=d['class_names'])
    overall_perf = pd.DataFrame(np.zeros(1), index=['accuracy'])
    for name in (d['class_names']):
        total_acc += df.loc[name, name]
        recall.loc[name] = df.loc[name, name] / df_actual.loc[name]
        precision.loc[name] = df.loc[name, name] / df_pred.loc[name]

    accuracy = total_acc / totals * 100
    error_rate = 100 - accuracy
    overall_perf.loc['accuracy', 0] = accuracy

    performance_test = {'accuracy': accuracy, 'error_rate': error_rate, 'recall': recall,
                        'precision': precision, 'overall_acc': overall_perf}

    print('RESULTS OF TEST {}'.format(d['model_name']))
    print("the accuracy of the model is {} %".format(performance_test['accuracy']))
    print("the error rate of the model is {} %".format(performance_test['error_rate']))
    print("recall : \n{}".format(performance_test['recall']))
    print("precision : \n{}".format(performance_test['precision']))

    return performance_test


class AverageMeter(object): #https://github.com/pranoyr/cnn-lstm/blob/7062a1214ca0dbb5ba07d8405f9fbcd133b1575e/utils.py#L52
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count  #
