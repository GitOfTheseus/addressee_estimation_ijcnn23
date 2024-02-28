import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from B_TRAIN.utils import join_slot, shuffle_dataset, create_eval_group, get_transformations
from B_TRAIN.custom_datasets import CustomDataset_vision, CustomDataset_pose

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

def data_loader_training(d, slot_folders, csv_training, test_filename):

    # if you already have the label files for the ten-fold cross validation you can comment
    # join_slot
    # shuffle_dataset
    # create_eval_group
    join_slot(model=d['model'], data_dir=d['data_dir'], label_filename=d['label_filename'],
             slot_test=d['slot_test'], slot_folders=slot_folders, train_filename=csv_training[0],
              test_filename=test_filename, training_complete=d['training_complete'])

    shuffle_dataset(model=d['model'], data_dir=d['data_dir'], label_filename=csv_training[0])

    create_eval_group(model=d['model'], data_dir=d['data_dir'], train_filename=csv_training[0], eval_filename=csv_training[1],
                      tot_sequences=30, class_names=d['class_names'])

    data_transform_face, data_transform_pose = get_transformations()

    if d['model'] == 'cnn_lstm_mm':
        dataset = {x: CustomDataset_vision(label_dir=os.path.join(d['data_dir'], csv_training[i]),
                                           root_dir=d['data_dir'], transform_face=data_transform_face[x],
                                           transform_pose=data_transform_pose[x], phase='train')
                   for i, x in enumerate(['train', 'eval'])}
    elif d['model'] == 'cnn_lstm_mm_latefusion':
        dataset = {x: CustomDataset_vision(label_dir=os.path.join(d['data_dir'], csv_training[i]),
                   root_dir=d['data_dir'], transform_face=data_transform_face[x],
                   transform_pose=data_transform_pose[x], phase='train')
               for i, x in enumerate(['train', 'eval'])}
    elif d['model'] == 'cnn_lstm_image':
        dataset = {x: CustomDataset_vision(label_dir=os.path.join(d['data_dir'], csv_training[i]),
                                           root_dir=d['data_dir'], transform_face=data_transform_face[x],
                                           transform_pose=data_transform_pose[x], phase='train')
                   for i, x in enumerate(['train', 'eval'])}
    elif d['model'] == 'cnn_lstm_pose':
        dataset = {x: CustomDataset_pose(label_dir=os.path.join(d['data_dir'], csv_training[i]),
                                         root_dir=d['data_dir'], transform_pose=data_transform_pose[x], phase='train')
                   for i, x in enumerate(['train', 'eval'])}

    dataloader = {x: DataLoader(dataset[x], batch_size=d['batch'], shuffle=False, num_workers=0, worker_init_fn=seed_worker,
    generator=g) for x in ['train', 'eval']}

    return dataloader


def data_loader_test(d, test_filename):

    data_transform_face, data_transform_pose = get_transformations()

    if d['model'] == 'cnn_lstm_mm':
        dataset = CustomDataset_vision(label_dir=os.path.join(d['data_dir'], test_filename),
                            root_dir=d['data_dir'], transform_face=data_transform_face['test'],
                            transform_pose=data_transform_pose['test'], phase='test')
    elif d['model'] == 'cnn_lstm_mm_latefusion':
        dataset = CustomDataset_vision(label_dir=os.path.join(d['data_dir'], test_filename),
                            root_dir=d['data_dir'], transform_face=data_transform_face['test'],
                            transform_pose=data_transform_pose['test'], phase='test')
    elif d['model'] == 'cnn_lstm_image':
        dataset = CustomDataset_vision(label_dir=os.path.join(d['data_dir'], test_filename),
                            root_dir=d['data_dir'], transform_face=data_transform_face['test'],
                            transform_pose=data_transform_pose['test'], phase='test')
    elif d['model'] == 'cnn_lstm_pose':
        dataset = CustomDataset_pose(label_dir=os.path.join(d['data_dir'], test_filename),
                            root_dir=d['data_dir'], transform_pose=data_transform_pose['test'], phase='test')

    dataloader = DataLoader(dataset, batch_size=d['batch'], shuffle=False, num_workers=0, worker_init_fn=seed_worker,
    generator=g)

    return dataloader


def data_loader_test_solo(d, test_filename):

    data_transform_face, data_transform_pose = get_transformations()


    if d['model'] == 'cnn_lstm_mm':
        dataset = CustomDataset_vision(label_dir=os.path.join(d['labels_dir'], test_filename),
                            root_dir=d['data_dir'], transform_face=data_transform_face['test'],
                            transform_pose=data_transform_pose['test'], phase='test')
    elif d['model'] == 'cnn_lstm_mm_latefusion':
        dataset = CustomDataset_vision(label_dir=os.path.join(d['data_dir'], test_filename),
                            root_dir=d['data_dir'], transform_face=data_transform_face['test'],
                            transform_pose=data_transform_pose['test'], phase='test')
    elif d['model'] == 'cnn_lstm_image':
        dataset = CustomDataset_vision(label_dir=os.path.join(d['labels_dir'], test_filename),
                            root_dir=d['data_dir'], transform_face=data_transform_face['test'],
                            transform_pose=data_transform_pose['test'], phase='test')
    elif d['model'] == 'cnn_lstm_pose':
        dataset = CustomDataset_pose(label_dir=os.path.join(d['labels_dir'], test_filename),
                            root_dir=d['data_dir'], transform_pose=data_transform_pose['test'], phase='test')

    dataloader = DataLoader(dataset, batch_size=d['batch'], shuffle=False, num_workers=0, worker_init_fn=seed_worker,
    generator=g)

    return dataloader
