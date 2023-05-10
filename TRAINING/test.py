import os.path

import pandas as pd
import torch
import numpy as np
from models import build_models
from utils import calculate_performance
from loaders import data_loader_test, data_loader_test_solo
from plots import plot_confusion_matrix


torch.manual_seed(0)
np.random.seed(0)

def test_vision(d, dataloader, cnn_image_d, cnn_pose_d, lstm_vision_d):

    device = d['device']
    models_list = [cnn_image_d, cnn_pose_d, lstm_vision_d]

    for i, model in enumerate(models_list):
        state = torch.load(d['model_dir_list'][i])
        model['model'].load_state_dict(state['state_dict'])
        model[d['optimizer1']].load_state_dict(state['optimizer_state_dict'])
        model['model'].eval()
    
    confusion_matrix = np.zeros((len(d['class_names']), len(d['class_names'])))
    correct = 0
    total = 0
    sequence_error_list = []
    label_error_list = []
    predictions_error_list = []
    all_labels = []
    all_predictions = []
    all_scores = []
    all_sequences = []
    all_intervals = []

    with torch.no_grad():
        for i, (images, poses, labels, sequence, intervals) in enumerate(dataloader):
            # calculate outputs by running images through the network
            interval_list = []
            images = images.to(device)
            poses = poses.to(device)
            labels = labels.to(device)

            face_tensor = cnn_image_d['model'](images)
            pose_tensor = cnn_pose_d['model'](poses)

            mm_tensor = torch.cat((face_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor,
                                           pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor,
                                           pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor,
                                           pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor,
                                           pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor), 1)

            output, current_batch = lstm_vision_d['model'](mm_tensor)
            _, preds = torch.max(output, 1)
            for ii, pred in enumerate(preds):
                print(pred, output[ii])

            for ss, prediction in enumerate(preds):
                score = output[ss, prediction]
                all_scores.append(round(score.item(),2))

            sequence = (list(dict.fromkeys(list(sequence))))

            for k, interval in enumerate(intervals):
                if (k+1)%10 == 0:
                    interval_list.append(interval)

            labels = labels.reshape(current_batch, d['seq_len'])
            labels = torch.transpose(labels, 0, 1)
            labels = labels[-1]
            total += labels.size(0)

            correct += (preds == labels).sum().item()
            for i, l in enumerate(labels):
                all_labels.append(labels[i].item())
                all_predictions.append(preds[i].item())
                all_sequences.append(sequence[i])
                all_intervals.append(interval_list[i])

                confusion_matrix[int(labels[i]), int(preds[i])] += 1
                if labels[i] != preds[i]:
                    sequence_error_list.append(sequence[i])
                    label_error_list.append(labels[i].item())
                    predictions_error_list.append(preds[i].item())

        errors = {'sequence': sequence_error_list, 'label': label_error_list, 'prediction': predictions_error_list}
        error_df = pd.DataFrame(errors)
        second_check = {'label': all_labels, 'predictions': all_predictions, 'score':all_scores, 'sequence': all_sequences, 'interval': all_intervals}
        intervals_check = pd.DataFrame(second_check)
    
    return confusion_matrix, error_df, intervals_check


def test_vision_with_other(d, dataloader, cnn_image_d, cnn_pose_d, lstm_vision_with_other_d):
    device = d['device']

    cnn_face_speaker_d = cnn_image_d
    cnn_pose_speaker_d = cnn_pose_d
    cnn_face_other_d = cnn_image_d
    cnn_pose_other_d= cnn_pose_d

    models_list = cnn_face_speaker_d, cnn_pose_speaker_d, cnn_face_other_d, cnn_pose_other_d, lstm_vision_with_other_d

    for i, model in enumerate(models_list):
        state = torch.load(d['model_dir_list'][i])
        model['model'].load_state_dict(state['state_dict'])
        model[d['optimizer1']].load_state_dict(state['optimizer_state_dict'])
        model['model'].eval()

    confusion_matrix = np.zeros((len(d['class_names']), len(d['class_names'])))
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (images_speaker, poses_speaker, images_other, poses_other, labels) in enumerate(dataloader):
            # calculate outputs by running images through the network

            images_speaker = images_speaker.to(device)
            poses_speaker = poses_speaker.to(device)
            images_other = images_other.to(device)
            poses_other = poses_other.to(device)
            labels = labels.to(device)

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
            total += labels.size(0)

            correct += (preds == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[int(labels[i]), int(preds[i])] += 1

    return confusion_matrix


def test_vision_audio(d, dataloader, cnn_image_d, cnn_pose_d, cnn_audio_d, lstm_vision_audio_d):
    device = d['device']
    models_list = [cnn_image_d, cnn_pose_d, cnn_audio_d, lstm_vision_audio_d]

    for i, model in enumerate(models_list):
        state = torch.load(d['model_dir_list'][i])
        model['model'].load_state_dict(state['state_dict'])
        model[d['optimizer1']].load_state_dict(state['optimizer_state_dict'])
        model['model'].eval()

    confusion_matrix = np.zeros((len(d['class_names']), len(d['class_names'])))
    correct = 0
    total = 0
    sequence_error_list = []
    label_error_list = []
    predictions_error_list = []
    all_labels = []
    all_predictions = []
    all_scores = []
    all_sequences = []
    all_intervals = []

    with torch.no_grad():
        for i, (images, poses, audios, labels, sequence, intervals) in enumerate(dataloader):
            # calculate outputs by running images through the network
            interval_list = []
            images = images.to(device)
            poses = poses.to(device)
            audios = audios.to(device)
            labels = labels.to(device)

            face_tensor = cnn_image_d['model'](images)
            pose_tensor = cnn_pose_d['model'](poses)
            audio_tensor = cnn_audio_d['model'](audios)

            mm_tensor = torch.cat(
                (face_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor,
                 pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor,
                 pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor,
                 pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor,
                 pose_tensor, pose_tensor, pose_tensor, pose_tensor, pose_tensor, audio_tensor), 1)

            output, current_batch = lstm_vision_audio_d['model'](mm_tensor)
            _, preds = torch.max(output, 1)

            for ss, prediction in enumerate(preds):
                score = output[ss, prediction]
                all_scores.append(round(score.item(),2))

            sequence = (list(dict.fromkeys(list(sequence))))
            for k, interval in enumerate(intervals):
                if (k + 1) % 10 == 0:
                    interval_list.append(interval)

            labels = labels.reshape(current_batch, d['seq_len'])
            labels = torch.transpose(labels, 0, 1)
            labels = labels[-1]
            total += labels.size(0)

            correct += (preds == labels).sum().item()
            for i, l in enumerate(labels):

                all_labels.append(labels[i].item())
                all_predictions.append(preds[i].item())
                all_sequences.append(sequence[i])
                all_intervals.append(interval_list[i])

                confusion_matrix[int(labels[i]), int(preds[i])] += 1
                if labels[i] != preds[i]:
                    sequence_error_list.append(sequence[i])
                    label_error_list.append(labels[i].item())
                    predictions_error_list.append(preds[i].item())

            errors = {'sequence': sequence_error_list, 'label': label_error_list, 'prediction': predictions_error_list}
            error_df = pd.DataFrame(errors)
            second_check = {'label': all_labels, 'predictions': all_predictions, 'score': all_scores,
                            'sequence': all_sequences, 'interval': all_intervals}
            intervals_check = pd.DataFrame(second_check)

    return confusion_matrix, error_df, intervals_check


def test_audio(d, dataloader, cnn_image_d, lstm_audio_d):

    device = d['device']
    models_list = [cnn_image_d, lstm_audio_d]
    
    for i, model in enumerate(models_list):
        state = torch.load(d['model_dir_list'][i])
        model['model'].load_state_dict(state['state_dict'])
        model[d['optimizer1']].load_state_dict(state['optimizer_state_dict'])
        model['model'].eval()
    
    confusion_matrix = np.zeros((len(d['class_names']), len(d['class_names'])))
    correct = 0
    total = 0
    sequence_error_list = []
    label_error_list = []
    predictions_error_list = []
    all_labels = []
    all_predictions = []
    all_scores = []
    all_sequences = []
    all_intervals = []
    
    with torch.no_grad():
        for i, (images, labels, sequence, intervals) in enumerate(dataloader):
            # calculate outputs by running images through the network
            interval_list = []
            images = images.to(device)
            labels = labels.to(device)

            tensor = cnn_image_d['model'](images)

            output, current_batch = lstm_audio_d['model'](tensor)
            _, preds = torch.max(output, 1)

            for ss, prediction in enumerate(preds):
                score = output[ss, prediction]
                all_scores.append(round(score.item(),2))

            sequence = (list(dict.fromkeys(list(sequence))))
            for k, interval in enumerate(intervals):
                if (k + 1) % 10 == 0:
                    interval_list.append(interval)

            labels = labels.reshape(current_batch, d['seq_len'])
            labels = torch.transpose(labels, 0, 1)
            labels = labels[-1]
            total += labels.size(0)

            correct += (preds == labels).sum().item()
            for i, l in enumerate(labels):

                all_labels.append(labels[i].item())
                all_predictions.append(preds[i].item())
                all_sequences.append(sequence[i])
                all_intervals.append(interval_list[i])

                confusion_matrix[int(labels[i]), int(preds[i])] += 1
                if labels[i] != preds[i]:
                    sequence_error_list.append(sequence[i])
                    label_error_list.append(labels[i].item())
                    predictions_error_list.append(preds[i].item())

            errors = {'sequence': sequence_error_list, 'label': label_error_list, 'prediction': predictions_error_list}
            error_df = pd.DataFrame(errors)
            second_check = {'label': all_labels, 'predictions': all_predictions, 'score': all_scores,
                            'sequence': all_sequences, 'interval': all_intervals}
            intervals_check = pd.DataFrame(second_check)
    
    return confusion_matrix, error_df, intervals_check


def test_image(d, dataloader, cnn_image_d, lstm_image_d):

    device = d['device']
    models_list = [cnn_image_d, lstm_image_d]
    
    for i, model in enumerate(models_list):
        state = torch.load(d['model_dir_list'][i])
        model['model'].load_state_dict(state['state_dict'])
        model[d['optimizer1']].load_state_dict(state['optimizer_state_dict'])
        model['model'].eval()
    
    confusion_matrix = np.zeros((len(d['class_names']), len(d['class_names'])))
    correct = 0
    total = 0
    sequence_error_list = []
    label_error_list = []
    predictions_error_list = []
    all_labels = []
    all_predictions = []
    all_scores = []
    all_sequences = []
    all_intervals = []

    with torch.no_grad():
        # AGGIUNTO SEQUENCE
        for i, (images, poses, labels, sequence, intervals) in enumerate(dataloader):
            # calculate outputs by running images through the network
            interval_list = []
            images = images.to(device)
            labels = labels.to(device)

            tensor = cnn_image_d['model'](images)

            output, current_batch = lstm_image_d['model'](tensor)
            _, preds = torch.max(output, 1)

            for ss, prediction in enumerate(preds):
                score = output[ss, prediction]
                all_scores.append(round(score.item(),2))

            #AGGIUNTO
            sequence = (list(dict.fromkeys(list(sequence))))
            for k, interval in enumerate(intervals):
                if (k + 1) % 10 == 0:
                    interval_list.append(interval)

            labels = labels.reshape(current_batch, d['seq_len'])
            labels = torch.transpose(labels, 0, 1)
            labels = labels[-1]
            total += labels.size(0)

            correct += (preds == labels).sum().item()

            for i, l in enumerate(labels):

                all_labels.append(labels[i].item())
                all_predictions.append(preds[i].item())
                all_sequences.append(sequence[i])
                all_intervals.append(interval_list[i])

                confusion_matrix[int(labels[i]), int(preds[i])] += 1
                if labels[i] != preds[i]:
                    sequence_error_list.append(sequence[i])
                    label_error_list.append(labels[i].item())
                    predictions_error_list.append(preds[i].item())

        errors = {'sequence': sequence_error_list, 'label': label_error_list, 'prediction': predictions_error_list}
        error_df = pd.DataFrame(errors)
        second_check = {'label': all_labels, 'predictions': all_predictions, 'score':all_scores, 'sequence': all_sequences, 'interval': all_intervals}
        intervals_check = pd.DataFrame(second_check)

    return confusion_matrix, error_df, intervals_check


def test_pose(d, dataloader, cnn_pose_d, lstm_pose_d):

    device = d['device']
    models_list = [cnn_pose_d, lstm_pose_d]
    
    for i, model in enumerate(models_list):
        state = torch.load(d['model_dir_list'][i])
        model['model'].load_state_dict(state['state_dict'])
        model[d['optimizer2']].load_state_dict(state['optimizer_state_dict'])
        model['model'].eval()
    
    confusion_matrix = np.zeros((len(d['class_names']), len(d['class_names'])))
    correct = 0
    total = 0
    sequence_error_list = []
    label_error_list = []
    predictions_error_list = []
    all_labels = []
    all_predictions = []
    all_scores = []
    all_sequences = []
    all_intervals = []
    
    with torch.no_grad():
        for i, (poses, labels, sequence, intervals) in enumerate(dataloader):
            # calculate outputs by running poses through the network
            interval_list = []
            poses = poses.to(device)
            labels = labels.to(device)

            tensor = cnn_pose_d['model'](poses)

            output, current_batch = lstm_pose_d['model'](tensor)
            _, preds = torch.max(output, 1)

            for ss, prediction in enumerate(preds):
                score = output[ss, prediction]
                all_scores.append(round(score.item(),2))

            # AGGIUNTO
            sequence = (list(dict.fromkeys(list(sequence))))
            for k, interval in enumerate(intervals):
                if (k + 1) % 10 == 0:
                    interval_list.append(interval)

            labels = labels.reshape(current_batch, d['seq_len'])
            labels = torch.transpose(labels, 0, 1)
            labels = labels[-1]
            total += labels.size(0)

            correct += (preds == labels).sum().item()
            for i, l in enumerate(labels):

                all_labels.append(labels[i].item())
                all_predictions.append(preds[i].item())
                all_sequences.append(sequence[i])
                all_intervals.append(interval_list[i])

                confusion_matrix[int(labels[i]), int(preds[i])] += 1
                if labels[i] != preds[i]:
                    sequence_error_list.append(sequence[i])
                    label_error_list.append(labels[i].item())
                    predictions_error_list.append(preds[i].item())

            errors = {'sequence': sequence_error_list, 'label': label_error_list, 'prediction': predictions_error_list}

            error_df = pd.DataFrame(errors)
            second_check = {'label': all_labels, 'predictions': all_predictions, 'score': all_scores,
                            'sequence': all_sequences, 'interval': all_intervals}
            intervals_check = pd.DataFrame(second_check)
    
    return confusion_matrix, error_df, intervals_check


def test_cnn_audio(d, dataloader, cnn_audio_d):

    device = d['device']
    models_list = [cnn_audio_d]

    for i, model in enumerate(models_list):
        state = torch.load(d['model_dir_list'][i])
        model['model'].load_state_dict(state['state_dict'])
        model[d['optimizer1']].load_state_dict(state['optimizer_state_dict'])
        model['model'].eval()

    confusion_matrix = np.zeros((len(d['class_names']), len(d['class_names'])))
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (spectra, labels) in enumerate(dataloader):
            # calculate outputs by running images through the network

            spectra = spectra.to(device)
            labels = labels.to(device)

            output, current_batch = cnn_audio_d['model'](spectra)

            _, preds = torch.max(output, 1)

            total += labels.size(0)

            correct += (preds == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[int(labels[i]), int(preds[i])] += 1

    return confusion_matrix


def test(dict_dir, test_filename):

    d = np.load(dict_dir, allow_pickle='TRUE').item()
    dataloader = data_loader_test(d, test_filename)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    d['device'] = device

    cnn_image_d, cnn_pose_d, cnn_audio_d, lstm_vision_d, lstm_vision_with_other_d, lstm_vision_audio_d, lstm_audio_d, lstm_image_d, lstm_pose_d = build_models(d)
    
    if d['model'] == 'cnn_lstm_mm':
        confusion_matrix, error_df, intervals_check = test_vision(d, dataloader, cnn_image_d, cnn_pose_d, lstm_vision_d)
    elif d['model'] == 'cnn_lstm_mm_with_other':
        confusion_matrix = test_vision_with_other(d, dataloader, cnn_image_d, cnn_pose_d, lstm_vision_with_other_d)
    elif d['model'] == 'cnn_lstm_vision_audio':
        confusion_matrix, error_df, intervals_check = test_vision_audio(d, dataloader, cnn_image_d, cnn_pose_d, cnn_image_d, lstm_vision_audio_d)
    elif d['model'] == 'cnn_lstm_audio':
        confusion_matrix, error_df = test_audio(d, dataloader, cnn_image_d, lstm_audio_d)
    elif d['model'] == 'cnn_lstm_image':
        confusion_matrix, error_df, intervals_check = test_image(d, dataloader, cnn_image_d, lstm_image_d)
    elif d['model'] == 'cnn_lstm_pose':
        confusion_matrix, error_df, intervals_check = test_pose(d, dataloader, cnn_pose_d, lstm_pose_d)
    elif d['model'] == 'cnn_audio':
        confusion_matrix = test_cnn_audio(d, dataloader, cnn_audio_d)

    d['confusion_matrix'] = confusion_matrix
    performance_test = calculate_performance(d)
    plot_confusion_matrix(d, performance_test)
    d['performance_test'] = performance_test

    np.save(dict_dir, d)
    print('Model INFO updated in {}_info.npy'.format(d['model']))
    print('\n\n')

    return d, error_df


def test_solo(dict_dir, test_filename, data_dir, labels_dir):
    d = np.load(dict_dir, allow_pickle='TRUE').item()
    d['data_dir'] = data_dir
    d['labels_dir'] = labels_dir
    dataloader = data_loader_test_solo(d, test_filename)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    d['device'] = device

    cnn_image_d, cnn_pose_d, cnn_audio_d, lstm_vision_d, lstm_vision_with_other_d, lstm_vision_audio_d, lstm_audio_d, lstm_image_d, lstm_pose_d = build_models(
        d)

    if d['model'] == 'cnn_lstm_mm':
        confusion_matrix, error_df, intervals_check_df = test_vision(d, dataloader, cnn_image_d, cnn_pose_d, lstm_vision_d)
    elif d['model'] == 'cnn_lstm_mm_with_other':
        confusion_matrix = test_vision_with_other(d, dataloader, cnn_image_d, cnn_pose_d, lstm_vision_with_other_d)
    elif d['model'] == 'cnn_lstm_vision_audio':
        confusion_matrix, error_df, intervals_check_df = test_vision_audio(d, dataloader, cnn_image_d, cnn_pose_d, cnn_image_d,
                                                       lstm_vision_audio_d)
    elif d['model'] == 'cnn_lstm_audio':
        confusion_matrix, error_df = test_audio(d, dataloader, cnn_image_d, lstm_audio_d)
    elif d['model'] == 'cnn_lstm_image':
        confusion_matrix, error_df, intervals_check_df = test_image(d, dataloader, cnn_image_d, lstm_image_d)
    elif d['model'] == 'cnn_lstm_pose':
        confusion_matrix, error_df, intervals_check_df = test_pose(d, dataloader, cnn_pose_d, lstm_pose_d)
    elif d['model'] == 'cnn_audio':
        confusion_matrix = test_cnn_audio(d, dataloader, cnn_audio_d)

    d['confusion_matrix'] = confusion_matrix
    performance_test = calculate_performance(d)
    plot_confusion_matrix(d, performance_test)
    d['performance_test'] = performance_test

    return d, error_df, intervals_check_df
