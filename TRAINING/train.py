import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
from torchvision import transforms, datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from datetime import datetime
import time
import os
import random
from models import build_models
from utils import AverageMeter
from plots import plot_performances


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
#torch.use_deterministic_algorithms(True)

def train_epoch_vision(d, dataloader, cnn_image_d, cnn_pose_d, lstm_vision_d, performances):

    ## AGGIUNTO ##
    # Early stopping
    last_loss = 100
    patience = d['patience']
    triggertimes = 0
    ## AGGIUNTO ##

    device = d['device']
    models_list = cnn_image_d, cnn_pose_d, lstm_vision_d

    for epoch in range(1, d['num_epochs'] + 1):

        for phase in ['train', 'eval']:
            if phase == 'train':
                for model in models_list:
                    model['model'].train()
            else:
                for model in models_list:
                    model['model'].eval()

            losses = AverageMeter()
            accuracies = AverageMeter()

            # Load in the data in batches using the train_loader object
            for i, (images, poses, labels, sequence, intervals) in enumerate(dataloader[phase]):
                # Move tensors to the configured device
                images = images.to(device)
                poses = poses.to(device)
                labels = labels.to(device)
                for model in models_list:
                    if model == cnn_image_d:
                        model[d['optimizer1']].zero_grad()
                    else:
                        model[d['optimizer2']].zero_grad()


                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass
                    face_tensor = cnn_image_d['model'](images)
                    pose_tensor = cnn_pose_d['model'](poses)

                    mm_tensor = torch.cat((face_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor,
                                           pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor,
                                           pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor,
                                           pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor,
                                           pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor), 1)

                    output, current_batch = lstm_vision_d['model'](mm_tensor)
                    _, preds = torch.max(output, 1)

                    labels = labels.reshape(current_batch, d['seq_len'])
                    labels = torch.transpose(labels, 0, 1)
                    labels = labels[-1]
                    loss = lstm_vision_d['criterion'](output, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        for model in models_list:
                            if model == cnn_image_d:
                                model[d['optimizer1']].step()
                            else:
                                model[d['optimizer2']].step()


                batch_accuracy = torch.sum(preds == labels.data) / current_batch
                losses.update(loss.item(), current_batch)
                accuracies.update(batch_accuracy.item(), current_batch)

            print('{} set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
                phase,
                len(dataloader[phase].dataset),
                losses.avg,
                accuracies.avg* 100))

            if phase == 'train':
                for model in models_list:
                    if model == cnn_image_d:
                        model['scheduler1'].step()
                    else:
                        model['scheduler2'].step()


                performances['train_loss'].append(losses.avg)
                performances['train_acc'].append(accuracies.avg)
            elif phase == 'eval':
                performances['eval_loss'].append(losses.avg)
                performances['eval_acc'].append(accuracies.avg)
                if accuracies.avg > performances['best_acc']:
                    performances['best_acc'] = accuracies.avg

        plot_performances(epoch, performances, d)

        ## AGGIUNTO ##
        # Early stopping
        if losses.avg > last_loss:
            trigger_times += 1
            print('Trigger Times:', trigger_times)
            print("Epoch {}\n".format(epoch))

            if trigger_times >= patience:

                print("Epoch {}\n".format(epoch))
                print('Early stopping!\nStart to test process.')
                for model in models_list:
                    if model == cnn_image_d:
                        model['state'] = {'epoch': epoch, 'state_dict': model['model'].state_dict(),
                                          'optimizer_state_dict': model[d['optimizer1']].state_dict(),
                                          'loss_function': model['criterion']}
                    else:
                        model['state'] = {'epoch': epoch, 'state_dict': model['model'].state_dict(),
                                          'optimizer_state_dict': model[d['optimizer2']].state_dict(),
                                          'loss_function': model['criterion']}

                return performances, models_list

        else:
            print('trigger times: 0')
            print("Epoch {}\n".format(epoch))
            trigger_times = 0

        last_loss = losses.avg
        ## AGGIUNTO ##


    for model in models_list:
        if model == cnn_image_d:
            model['state'] = {'epoch': epoch, 'state_dict': model['model'].state_dict(),
                              'optimizer_state_dict': model[d['optimizer1']].state_dict(),
                              'loss_function': model['criterion']}
        else:
            model['state'] = {'epoch': epoch, 'state_dict': model['model'].state_dict(),
                              'optimizer_state_dict': model[d['optimizer2']].state_dict(),
                              'loss_function': model['criterion']}

    return performances, models_list


def train_epoch_vision_with_other(d, dataloader, cnn_image_d, cnn_pose_d, lstm_vision_with_other_d, performances):

    ## AGGIUNTO ##
    # Early stopping
    last_loss = 100
    patience = d['patience']
    triggertimes = 0
    ## AGGIUNTO ##

    device = d['device']
    cnn_face_speaker_d = cnn_image_d
    cnn_pose_speaker_d = cnn_pose_d
    cnn_face_other_d = cnn_image_d
    cnn_pose_other_d= cnn_pose_d

    models_list = cnn_face_speaker_d, cnn_pose_speaker_d, cnn_face_other_d, cnn_pose_other_d, lstm_vision_with_other_d

    for epoch in range(1, d['num_epochs'] + 1):

        for phase in ['train', 'eval']:
            if phase == 'train':
                for model in models_list:
                    model['model'].train()
            else:
                for model in models_list:
                    model['model'].eval()

            losses = AverageMeter()
            accuracies = AverageMeter()

            # Load in the data in batches using the train_loader object
            for i, (images_speaker, poses_speaker, images_other, poses_other, labels) in enumerate(dataloader[phase]):
                # Move tensors to the configured device
                images_speaker = images_speaker.to(device)
                poses_speaker = poses_speaker.to(device)
                images_other = images_other.to(device)
                poses_other = poses_other.to(device)
                labels = labels.to(device)
                for model in models_list:
                    if model == cnn_face_speaker_d or cnn_face_other_d:
                        model[d['optimizer1']].zero_grad()
                    else:
                        model[d['optimizer2']].zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass
                    face_s_tensor = cnn_face_speaker_d['model'](images_speaker)
                    pose_s_tensor = cnn_pose_speaker_d['model'](poses_speaker)

                    face_o_tensor = cnn_face_other_d['model'](images_other)
                    pose_o_tensor = cnn_pose_other_d['model'](poses_other)

                    mm_tensor_speaker = torch.cat((face_s_tensor, pose_s_tensor, pose_s_tensor, pose_s_tensor,
                                                   pose_s_tensor, pose_s_tensor, pose_s_tensor, pose_s_tensor,
                                                   pose_s_tensor, pose_s_tensor, pose_s_tensor, pose_s_tensor,
                                                   pose_s_tensor), 1)
                    mm_tensor_other = torch.cat((face_o_tensor, pose_o_tensor, pose_o_tensor, pose_o_tensor,
                                                 pose_o_tensor, pose_o_tensor, pose_o_tensor, pose_o_tensor,
                                                 pose_o_tensor, pose_o_tensor, pose_o_tensor, pose_o_tensor,
                                                 pose_o_tensor), 1)
                    mm_tensor = torch.cat((mm_tensor_speaker, mm_tensor_other), 1)

                    output, current_batch = lstm_vision_with_other_d['model'](mm_tensor)
                    _, preds = torch.max(output, 1)

                    labels = labels.reshape(current_batch, d['seq_len'])
                    labels = torch.transpose(labels, 0, 1)
                    labels = labels[-1]
                    loss = lstm_vision_with_other_d['criterion'](output, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        for model in models_list:
                            if model == cnn_face_speaker_d or cnn_face_other_d:
                                model[d['optimizer1']].step()
                            else:
                                model[d['optimizer2']].step()


                batch_accuracy = torch.sum(preds == labels.data) / current_batch
                losses.update(loss.item(), current_batch)
                accuracies.update(batch_accuracy.item(), current_batch)

            print('{} set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
                phase,
                len(dataloader[phase].dataset),
                losses.avg,
                accuracies.avg* 100))

            if phase == 'train':
                for model in models_list:
                    if model == cnn_face_speaker_d or cnn_face_other_d:
                        model['scheduler1'].step()
                    else:
                        model['scheduler2'].step()


                performances['train_loss'].append(losses.avg)
                performances['train_acc'].append(accuracies.avg)
            elif phase == 'eval':
                performances['eval_loss'].append(losses.avg)
                performances['eval_acc'].append(accuracies.avg)
                if accuracies.avg > performances['best_acc']:
                    performances['best_acc'] = accuracies.avg

        plot_performances(epoch, performances, d)

        ## AGGIUNTO ##
        # Early stopping
        if losses.avg > last_loss:
            trigger_times += 1
            print('Trigger Times:', trigger_times)
            print("Epoch {}\n".format(epoch))

            if trigger_times >= patience:

                print("Epoch {}\n".format(epoch))
                print('Early stopping!\nStart to test process.')
                for model in models_list:
                    if model == cnn_face_speaker_d or cnn_face_other_d:
                        model['state'] = {'epoch': epoch, 'state_dict': model['model'].state_dict(),
                                          'optimizer_state_dict': model[d['optimizer1']].state_dict(),
                                          'loss_function': model['criterion']}
                    else:
                        model['state'] = {'epoch': epoch, 'state_dict': model['model'].state_dict(),
                                          'optimizer_state_dict': model[d['optimizer2']].state_dict(),
                                          'loss_function': model['criterion']}

                return performances, models_list

        else:
            print('trigger times: 0')
            print("Epoch {}\n".format(epoch))
            trigger_times = 0

        last_loss = losses.avg
        ## AGGIUNTO ##


    for model in models_list:
        if model == cnn_face_speaker_d or cnn_face_other_d:
            model['state'] = {'epoch': epoch, 'state_dict': model['model'].state_dict(),
                              'optimizer_state_dict': model[d['optimizer1']].state_dict(),
                              'loss_function': model['criterion']}
        else:
            model['state'] = {'epoch': epoch, 'state_dict': model['model'].state_dict(),
                              'optimizer_state_dict': model[d['optimizer2']].state_dict(),
                              'loss_function': model['criterion']}

    return performances, models_list


def train_epoch_vision_audio(d, dataloader, cnn_image_d, cnn_pose_d, cnn_audio_d, lstm_vision_audio_d, performances):

    ## AGGIUNTO ##
    # Early stopping
    last_loss = 100
    patience = d['patience']
    triggertimes = 0
    ## AGGIUNTO ##

    device = d['device']
    models_list = cnn_image_d, cnn_pose_d, cnn_audio_d, lstm_vision_audio_d

    for epoch in range(1, d['num_epochs'] + 1):

        for phase in ['train', 'eval']:
            if phase == 'train':
                for model in models_list:
                    model['model'].train()
            else:
                for model in models_list:
                    model['model'].eval()

            losses = AverageMeter()
            accuracies = AverageMeter()

            # Load in the data in batches using the train_loader object
            for i, (images, poses, audios, labels, sequence, intervals) in enumerate(dataloader[phase]):
                # Move tensors to the configured device
                images = images.to(device)
                poses = poses.to(device)
                audios = audios.to(device)
                labels = labels.to(device)
                for model in models_list:
                    if model == cnn_image_d:
                        model[d['optimizer1']].zero_grad()
                    else:
                        model[d['optimizer2']].zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass
                    face_tensor = cnn_image_d['model'](images)
                    pose_tensor = cnn_pose_d['model'](poses)
                    audio_tensor = cnn_audio_d['model'](audios)

                    mm_tensor = torch.cat((face_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor,
                                           pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor,
                                           pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor,
                                           pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor,
                                           pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor, audio_tensor), 1)

                    output, current_batch = lstm_vision_audio_d['model'](mm_tensor)
                    _, preds = torch.max(output, 1)

                    labels = labels.reshape(current_batch, d['seq_len'])
                    labels = torch.transpose(labels, 0, 1)
                    labels = labels[-1]
                    loss = lstm_vision_audio_d['criterion'](output, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        for model in models_list:
                            if model == cnn_image_d:
                                model[d['optimizer1']].step()
                            else:
                                model[d['optimizer2']].step()


                batch_accuracy = torch.sum(preds == labels.data) / current_batch
                losses.update(loss.item(), current_batch)
                accuracies.update(batch_accuracy.item(), current_batch)

            print('{} set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
                phase,
                len(dataloader[phase].dataset),
                losses.avg,
                accuracies.avg* 100))

            if phase == 'train':
                for model in models_list:
                    if model == cnn_image_d:
                        model['scheduler1'].step()
                    else:
                        model['scheduler2'].step()


                performances['train_loss'].append(losses.avg)
                performances['train_acc'].append(accuracies.avg)
            elif phase == 'eval':
                performances['eval_loss'].append(losses.avg)
                performances['eval_acc'].append(accuracies.avg)
                if accuracies.avg > performances['best_acc']:
                    performances['best_acc'] = accuracies.avg

        plot_performances(epoch, performances, d)

        ## AGGIUNTO ##
        # Early stopping
        if losses.avg > last_loss:
            trigger_times += 1
            print('Trigger Times:', trigger_times)
            print("Epoch {}\n".format(epoch))

            if trigger_times >= patience:

                print("Epoch {}\n".format(epoch))
                print('Early stopping!\nStart to test process.')
                for model in models_list:
                    if model == cnn_image_d:
                        model['state'] = {'epoch': epoch, 'state_dict': model['model'].state_dict(),
                                          'optimizer_state_dict': model[d['optimizer1']].state_dict(),
                                          'loss_function': model['criterion']}
                    else:
                        model['state'] = {'epoch': epoch, 'state_dict': model['model'].state_dict(),
                                          'optimizer_state_dict': model[d['optimizer2']].state_dict(),
                                          'loss_function': model['criterion']}

                return performances, models_list

        else:
            print('trigger times: 0')
            print("Epoch {}\n".format(epoch))
            trigger_times = 0

        last_loss = losses.avg
        ## AGGIUNTO ##


    for model in models_list:
        if model == cnn_image_d:
            model['state'] = {'epoch': epoch, 'state_dict': model['model'].state_dict(),
                              'optimizer_state_dict': model[d['optimizer1']].state_dict(),
                              'loss_function': model['criterion']}
        else:
            model['state'] = {'epoch': epoch, 'state_dict': model['model'].state_dict(),
                              'optimizer_state_dict': model[d['optimizer2']].state_dict(),
                              'loss_function': model['criterion']}

    return performances, models_list


def train_epoch_audio(d, dataloader, cnn_image_d, lstm_audio_d, performances):

    ## AGGIUNTO ##
    # Early stopping
    last_loss = 100
    patience = d['patience']
    triggertimes = 0
    ## AGGIUNTO ##
    device = d['device']
    models_list = cnn_image_d, lstm_audio_d

    for epoch in range(1, d['num_epochs'] + 1):

        for phase in ['train', 'eval']:
            if phase == 'train':
                for model in models_list:
                    model['model'].train()
            else:
                for model in models_list:
                    model['model'].eval()

            losses = AverageMeter()
            accuracies = AverageMeter()

            # Load in the data in batches using the train_loader object
            for i, (images, labels, sequences) in enumerate(dataloader[phase]):

                # Move tensors to the configured device
                images = images.to(device)
                labels = labels.to(device)

                for model in models_list:
                    if model == cnn_image_d:
                        model[d['optimizer1']].zero_grad()
                    else:
                        model[d['optimizer2']].zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass
                    spect_tensor = cnn_image_d['model'](images)

                    output, current_batch = lstm_audio_d['model'](spect_tensor)
                    _, preds = torch.max(output, 1)

                    labels = labels.reshape(current_batch, d['seq_len'])
                    labels = torch.transpose(labels, 0, 1)
                    labels = labels[-1]
                    loss = lstm_audio_d['criterion'](output, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        for model in models_list:
                            if model == cnn_image_d:
                                model[d['optimizer1']].step()
                            else:
                                model[d['optimizer2']].step()

                batch_accuracy = torch.sum(preds == labels.data) / current_batch
                losses.update(loss.item(), current_batch)
                accuracies.update(batch_accuracy.item(), current_batch)

            print('{} set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
                phase,
                len(dataloader[phase].dataset),
                losses.avg,
                accuracies.avg* 100))

            if phase == 'train':
                for model in models_list:
                    if model == cnn_image_d:
                        model['scheduler1'].step()
                    else:
                        model['scheduler2'].step()
                performances['train_loss'].append(losses.avg)
                performances['train_acc'].append(accuracies.avg)
            elif phase == 'eval':
                performances['eval_loss'].append(losses.avg)
                performances['eval_acc'].append(accuracies.avg)
                if accuracies.avg > performances['best_acc']:
                    performances['best_acc'] = accuracies.avg

        plot_performances(epoch, performances, d)

        ## AGGIUNTO ##
        # Early stopping
        if losses.avg > last_loss:
            trigger_times += 1
            print('Trigger Times:', trigger_times)
            print("Epoch {}\n".format(epoch))

            if trigger_times >= patience:

                print("Epoch {}\n".format(epoch))
                print('Early stopping!\nStart to test process.')
                for model in models_list:
                    if model == cnn_image_d:
                        model['state'] = {'epoch': epoch, 'state_dict': model['model'].state_dict(),
                                          'optimizer_state_dict': model[d['optimizer1']].state_dict(),
                                          'loss_function': model['criterion']}
                    else:
                        model['state'] = {'epoch': epoch, 'state_dict': model['model'].state_dict(),
                                          'optimizer_state_dict': model[d['optimizer2']].state_dict(),
                                          'loss_function': model['criterion']}
                return performances, models_list

        else:
            print('trigger times: 0')
            print("Epoch {}\n".format(epoch))
            trigger_times = 0

        last_loss = losses.avg
        ## AGGIUNTO ##

    for model in models_list:
        if model == cnn_image_d:
            model['state'] = {'epoch': epoch, 'state_dict': model['model'].state_dict(),
                              'optimizer_state_dict': model[d['optimizer1']].state_dict(),
                              'loss_function': model['criterion']}
        else:
            model['state'] = {'epoch': epoch, 'state_dict': model['model'].state_dict(),
                              'optimizer_state_dict': model[d['optimizer2']].state_dict(),
                              'loss_function': model['criterion']}

    return performances, models_list
    
 
def train_epoch_image(d, dataloader, cnn_image_d, lstm_image_d, performances):

    ## AGGIUNTO ##
    # Early stopping
    last_loss = 100
    patience = d['patience']
    triggertimes = 0
    ## AGGIUNTO ##
    device = d['device']
    models_list = cnn_image_d, lstm_image_d

    for epoch in range(1, d['num_epochs'] + 1):

        for phase in ['train', 'eval']:
            if phase == 'train':
                for model in models_list:
                    model['model'].train()
            else:
                for model in models_list:
                    model['model'].eval()

            losses = AverageMeter()
            accuracies = AverageMeter()

            # Load in the data in batches using the train_loader object
            for i, (images, poses, labels, sequence, intervals) in enumerate(dataloader[phase]):
                # Move tensors to the configured device
                images = images.to(device)
                labels = labels.to(device)
                for model in models_list:
                    if model == cnn_image_d:
                        model[d['optimizer1']].zero_grad()
                    else:
                        model[d['optimizer2']].zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass
                    face_tensor = cnn_image_d['model'](images)

                    output, current_batch = lstm_image_d['model'](face_tensor)
                    _, preds = torch.max(output, 1)

                    labels = labels.reshape(current_batch, d['seq_len'])
                    labels = torch.transpose(labels, 0, 1)
                    labels = labels[-1]
                    loss = lstm_image_d['criterion'](output, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        for model in models_list:
                            if model == cnn_image_d:
                                model[d['optimizer1']].step()
                            else:
                                model[d['optimizer2']].step()

                batch_accuracy = torch.sum(preds == labels.data) / current_batch
                losses.update(loss.item(), current_batch)
                accuracies.update(batch_accuracy.item(), current_batch)

            print('{} set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
                phase,
                len(dataloader[phase].dataset),
                losses.avg,
                accuracies.avg* 100))

            if phase == 'train':
                for model in models_list:
                    if model == cnn_image_d:
                        model['scheduler1'].step()
                    else:
                        model['scheduler2'].step()

                performances['train_loss'].append(losses.avg)
                performances['train_acc'].append(accuracies.avg)
            elif phase == 'eval':
                performances['eval_loss'].append(losses.avg)
                performances['eval_acc'].append(accuracies.avg)
                if accuracies.avg > performances['best_acc']:
                    performances['best_acc'] = accuracies.avg

        plot_performances(epoch, performances, d)

        ## AGGIUNTO ##
        # Early stopping
        if losses.avg > last_loss:
            trigger_times += 1
            print('Trigger Times:', trigger_times)
            print("Epoch {}\n".format(epoch))
            if trigger_times >= patience:

                print("Epoch {}\n".format(epoch))
                print('Early stopping!\nStart to test process.')
                for model in models_list:
                    if model == cnn_image_d:
                        model['state'] = {'epoch': epoch, 'state_dict': model['model'].state_dict(),
                                          'optimizer_state_dict': model[d['optimizer1']].state_dict(),
                                          'loss_function': model['criterion']}
                    else:
                        model['state'] = {'epoch': epoch, 'state_dict': model['model'].state_dict(),
                                          'optimizer_state_dict': model[d['optimizer2']].state_dict(),
                                          'loss_function': model['criterion']}

                return performances, models_list

        else:
            print('trigger times: 0')
            print("Epoch {}\n".format(epoch))
            trigger_times = 0

        last_loss = losses.avg
        ## AGGIUNTO ##

    for model in models_list:
        if model == cnn_image_d:
            model['state'] = {'epoch': epoch, 'state_dict': model['model'].state_dict(),
                              'optimizer_state_dict': model[d['optimizer1']].state_dict(),
                              'loss_function': model['criterion']}
        else:
            model['state'] = {'epoch': epoch, 'state_dict': model['model'].state_dict(),
                              'optimizer_state_dict': model[d['optimizer2']].state_dict(),
                              'loss_function': model['criterion']}

    return performances, models_list


def train_epoch_pose(d, dataloader, cnn_pose_d, lstm_pose_d, performances):

    ## AGGIUNTO ##
    # Early stopping
    last_loss = 100
    patience = d['patience']
    triggertimes = 0
    ## AGGIUNTO ##
    device = d['device']
    models_list = cnn_pose_d, lstm_pose_d

    for epoch in range(1, d['num_epochs'] + 1):

        for phase in ['train', 'eval']:
            if phase == 'train':
                for model in models_list:
                    model['model'].train()

            else:
                for model in models_list:
                    model['model'].eval()

            losses = AverageMeter()
            accuracies = AverageMeter()

            # Load in the data in batches using the train_loader object
            for i, (poses, labels, sequence, intervals) in enumerate(dataloader[phase]):
                # Move tensors to the configured device
                poses = poses.to(device)
                labels = labels.to(device)
                for model in models_list:
                    model[d['optimizer2']].zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass
                    pose_tensor = cnn_pose_d['model'](poses)

                    output, current_batch = lstm_pose_d['model'](pose_tensor)
                    _, preds = torch.max(output, 1)

                    labels = labels.reshape(current_batch, d['seq_len'])
                    labels = torch.transpose(labels, 0, 1)
                    labels = labels[-1]
                    loss = lstm_pose_d['criterion'](output, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        for model in models_list:
                            model[d['optimizer2']].step()

                #print(current_batch)
                batch_accuracy = torch.sum(preds == labels.data) / current_batch
                losses.update(loss.item(), current_batch)
                accuracies.update(batch_accuracy.item(), current_batch)

            print('{} set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
                phase,
                len(dataloader[phase].dataset),
                losses.avg,
                accuracies.avg* 100))

            if phase == 'train':
                for model in models_list:
                    model['scheduler2'].step()

                performances['train_loss'].append(losses.avg)
                performances['train_acc'].append(accuracies.avg)
            elif phase == 'eval':
                performances['eval_loss'].append(losses.avg)
                performances['eval_acc'].append(accuracies.avg)
                if accuracies.avg > performances['best_acc']:
                    performances['best_acc'] = accuracies.avg

        plot_performances(epoch, performances, d)

        ## AGGIUNTO ##
        # Early stopping
        if losses.avg > last_loss:
            trigger_times += 1
            print('Trigger Times:', trigger_times)
            print("Epoch {}\n".format(epoch))

            if trigger_times >= patience:

                print("Epoch {}\n".format(epoch))
                print('Early stopping!\nStart to test process.')
                for model in models_list:
                    model['state'] = {'epoch': epoch, 'state_dict': model['model'].state_dict(),
                                      'optimizer_state_dict': model[d['optimizer2']].state_dict(),
                                      'loss_function': model['criterion']}

                return performances, models_list

        else:
            print('trigger times: 0')
            print("Epoch {}\n".format(epoch))
            trigger_times = 0

        last_loss = losses.avg
        ## AGGIUNTO ##

    for model in models_list:
        model['state'] = {'epoch': epoch, 'state_dict': model['model'].state_dict(),
                          'optimizer_state_dict': model[d['optimizer2']].state_dict(),
                          'loss_function': model['criterion']}

    return performances, models_list


def train_epoch_cnn_audio(d, dataloader, cnn_audio_d, performances):

    ## AGGIUNTO ##
    # Early stopping
    last_loss = 100
    patience = d['patience']
    triggertimes = 0
    ## AGGIUNTO ##
    device = d['device']
    models_list = [cnn_audio_d]

    for epoch in range(1, d['num_epochs'] + 1):

        for phase in ['train', 'eval']:
            if phase == 'train':
                for model in models_list:
                    model['model'].train()
            else:
                for model in models_list:
                    model['model'].eval()

            losses = AverageMeter()
            accuracies = AverageMeter()

            # Load in the data in batches using the train_loader object
            for i, (spectra, labels) in enumerate(dataloader[phase]):
                # Move tensors to the configured device
                spectra = spectra.to(device)
                labels = labels.to(device)
                for model in models_list:
                    model[d['optimizer1']].zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass
                    output, current_batch = cnn_audio_d['model'](spectra)

                    _, preds = torch.max(output, 1)

                    loss = cnn_audio_d['criterion'](output, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        for model in models_list:
                            model[d['optimizer1']].step()


                batch_accuracy = torch.sum(preds == labels.data) / current_batch * 100
                losses.update(loss.item(), current_batch)
                accuracies.update(batch_accuracy.item(), current_batch)


            print('{} set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
                phase,
                len(dataloader[phase].dataset),
                losses.avg,
                accuracies.avg))

            if phase == 'train':
                for model in models_list:
                    model['scheduler1'].step()

                performances['train_loss'].append(losses.avg)
                performances['train_acc'].append(accuracies.avg)
            elif phase == 'eval':
                performances['eval_loss'].append(losses.avg)
                performances['eval_acc'].append(accuracies.avg)
                if accuracies.avg > performances['best_acc']:
                    performances['best_acc'] = accuracies.avg

        plot_performances(epoch, performances, d)

        ## AGGIUNTO ##
        # Early stopping
        if losses.avg > last_loss:
            trigger_times += 1
            print('trigger times:', trigger_times)
            print("Epoch {}\n".format(epoch))

            if trigger_times >= patience:

                print('Early stopping!\nStart to test process.')
                for model in models_list:
                    model['state'] = {'epoch': epoch, 'state_dict': model['model'].state_dict(),
                                      'optimizer_state_dict': model[d['optimizer1']].state_dict(),
                                      'loss_function': model['criterion']}

                return performances, models_list

        else:
            print('trigger times: 0')
            print("Epoch {}\n".format(epoch))
            trigger_times = 0

        last_loss = losses.avg
        ## AGGIUNTO ##

    for model in models_list:
        model['state'] = {'epoch': epoch, 'state_dict': model['model'].state_dict(),
                          'optimizer_state_dict': model[d['optimizer1']].state_dict(),
                          'loss_function': model['criterion']}

    return performances, models_list


def train(d, dataloader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    d['device'] = device

    cnn_image_d, cnn_pose_d, cnn_audio_d, lstm_vision_d, lstm_vision_with_other_d, lstm_vision_audio_d, lstm_audio_d, lstm_image_d, lstm_pose_d = build_models(d)

    performances = {
        'best_acc': 0,
        'train_acc': [],
        'train_loss': [],
        'eval_loss': [],
        'eval_acc': [],
    }

    since = time.time()
    if d['model'] == 'cnn_lstm_mm':
        performances, models_list = train_epoch_vision(d, dataloader, cnn_image_d, cnn_pose_d, lstm_vision_d, performances)
    elif d['model'] == 'cnn_lstm_mm_with_other':
        performances, models_list = train_epoch_vision_with_other(d, dataloader, cnn_image_d, cnn_pose_d, lstm_vision_with_other_d,
                                                                  performances)
    elif d['model'] == 'cnn_lstm_vision_audio':
        performances, models_list = train_epoch_vision_audio(d, dataloader, cnn_image_d, cnn_pose_d, cnn_image_d, lstm_vision_audio_d, performances)
    elif d['model'] == 'cnn_lstm_audio':
        performances, models_list = train_epoch_audio(d, dataloader, cnn_image_d, lstm_audio_d, performances)
    elif d['model'] == 'cnn_lstm_image':
        performances, models_list = train_epoch_image(d, dataloader, cnn_image_d, lstm_image_d, performances)
    elif d['model'] == 'cnn_lstm_pose':
        performances, models_list = train_epoch_pose(d, dataloader, cnn_pose_d, lstm_pose_d, performances)
    elif d['model'] == 'cnn_audio':
        performances, models_list = train_epoch_cnn_audio(d, dataloader, cnn_audio_d, performances)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best eval Acc: {:4f}'.format(performances['best_acc']))

    return performances, models_list



