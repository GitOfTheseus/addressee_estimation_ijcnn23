import pandas as pd
import numpy as np
import os
import warnings

# general path of the dataset
dataset_path = ''
# n. slot of the dataset
slot = '09'
slot_path = os.path.join(dataset_path, slot)
label_path = os.path.join(slot_path, 'labels')
file_load_label = os.path.join(label_path, "all_labels.csv")
labels = pd.read_csv(file_load_label, sep='\t', index_col="index", na_values=[''])

chunk_size = 10   # n short intervals aggregated together

tot_long_intervals = int(labels['N_INTERVAL'].loc[labels.index[-1]])
n_interval = np.unique(labels['N_INTERVAL']).astype(int)
tot_frames = labels.index.size
lstm_df = pd.DataFrame()

if labels['N_INTERVAL'].any() > tot_long_intervals:
    warnings.warn("there are N_INTERVAL > than the N_INTERVAL at the last index of the dataframe, check how the df is sorted\n")

def divide_in_sequences(i_int):

    indexes = labels.loc[labels['N_INTERVAL'] == i_int].index

    start = indexes[0]
    end = indexes[-1] + 1

    all_lists = []
    for x in range(start, end, chunk_size):

        if x + chunk_size < end:
            index_list = list(range(x, x + chunk_size))
            all_lists.append(index_list)

        else:
            index_list = list(range(x, end))
            diff = chunk_size - len(index_list)

            if diff < chunk_size * 50 / 100:
                if indexes.size >= chunk_size:
                    index_list = list(range(x - diff, end))

                    all_lists.append(index_list)

                else:
                    for i in range(2, diff + 2):
                        index_list.append(end - i)
                    all_lists.append(index_list)

            else:
                index_list = []

    return all_lists


def check_missing_data():

    speaker = np.unique(labels['N_INTERVAL'].loc[labels[labels['FACE_SPEAKER'] == "missing"].index].to_numpy())
    other = np.unique(labels['N_INTERVAL'].loc[labels[labels['FACE_OTHER'] == "missing"].index].to_numpy())
    int_missing_data = np.unique(np.concatenate((speaker, other), axis=None))
    print("missing data: ", int_missing_data)

    return int_missing_data


lstm_labels = []
lstm_add = []
lstm_sequence = []
lstm_slot = []
lstm_speaker = []
lstm_n_int = []
lstm_start = []
lstm_end = []
lstm_frame = []
lstm_duration = []
lstm_img = []
lstm_pose_s = []
lstm_pose_o = []
lstm_face_s = []
lstm_face_o = []
lstm_audio = []
lstm_mel = []
lstm_mfcc = []

lstm_long_int = []


lstm_i = 0

int_missing_data = check_missing_data()

n_sequence = 0
for i_int in n_interval:

    all_lists = divide_in_sequences(i_int)

    for index_list in all_lists:

        if i_int not in int_missing_data:
            for i in index_list:
                lstm_labels.append(labels.loc[i, 'LABEL'])
                lstm_add.append(labels.loc[i, 'ADDRESSEE'])
                lstm_sequence.append(str(n_sequence))
                lstm_slot.append(slot)
                lstm_speaker.append(labels.loc[i, 'SPEAKER'])
                lstm_n_int.append(labels.loc[i, 'N_INTERVAL'])
                lstm_start.append(labels.loc[i, 'START_MSEC'])
                lstm_end.append(labels.loc[i, 'END_MSEC'])
                lstm_frame.append(labels.loc[i, 'FRAME_MSEC'])
                lstm_duration.append(labels.loc[i, 'DURATION_MSEC'])
                lstm_img.append(labels.loc[i, 'IMG'])
                lstm_pose_s.append(labels.loc[i, 'POSE_SPEAKER'])
                lstm_pose_o.append(labels.loc[i, 'POSE_OTHER'])
                lstm_face_s.append(labels.loc[i, 'FACE_SPEAKER'])
                lstm_face_o.append(labels.loc[i, 'FACE_OTHER'])
                lstm_audio.append(labels.loc[i, 'AUDIO'])
                lstm_mel.append(labels.loc[i, 'MEL_SPECT'])
                lstm_mfcc.append(labels.loc[i, 'MFCC_FEAT'])
            n_sequence += 1
    all_lists = []

if "missing" in lstm_face_s:
    print("error occurred")

#print(lstm_labels)

d = {'LABEL': lstm_labels, 'ADDRESSEE': lstm_add, 'SEQUENCE': lstm_sequence, 'SLOT': lstm_slot, 'SPEAKER': lstm_speaker, 'N_INTERVAL': lstm_n_int,
     'START_MSEC': lstm_start, 'END_MSEC': lstm_end, 'FRAME_MSEC': lstm_frame, 'DURATION_MSEC': lstm_duration,
     'IMG': lstm_img, 'POSE_SPEAKER': lstm_pose_s, 'FACE_SPEAKER': lstm_face_s, 'POSE_OTHER': lstm_pose_o,
     'FACE_OTHER': lstm_face_o, 'AUDIO': lstm_audio, 'MEL_SPECT': lstm_mel, 'MFCC_FEAT': lstm_mfcc}
lstm_labels_df = pd.DataFrame(data=d)
print(lstm_labels_df)

save_dir = os.path.join(label_path, "lstm_label_no_doubles.csv")

lstm_labels_df.to_csv(save_dir, sep='\t', index_label='index')



