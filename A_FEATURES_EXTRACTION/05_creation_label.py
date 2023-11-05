import pandas as pd
import numpy as np
import os
import logging
import warnings

# general path of the dataset
dataset_path = ''
# n. slot of the dataset
slot = '09'
slot_path = os.path.join(dataset_path, slot)
label_path = os.path.join(slot_path, 'labels')
img_label_file = f'labels_video_slot{slot}.csv'
img_label_dir = os.path.join(label_path, img_label_file)
audio_label_file = f'labels_audio_slot{slot}.csv'
audio_label_dir = os.path.join(label_path, audio_label_file)

img_path = os.path.join(slot_path, 'img')
img_list = os.listdir(img_path)
img_list.sort()

audio_path = os.path.join(slot_path, 'audio')
audio_list = os.listdir(audio_path)
audio_list.sort()
pose_speaker_path = os.path.join(slot_path, 'pose', 'speaker')
pose_speaker_list = os.listdir(pose_speaker_path)

pose_other_path = os.path.join(slot_path, 'pose', 'other')
pose_other_list = os.listdir(pose_other_path)

face_speaker_path = os.path.join(slot_path, 'face', 'speaker')
face_speaker_list = os.listdir(face_speaker_path)
face_speaker_not_detected = 0

face_other_path = os.path.join(slot_path, 'face', 'other')
face_other_list = os.listdir(face_other_path)
face_other_not_detected = 0

melspect_path = os.path.join(slot_path, 'audio_features', 'mel_spectrogram')
melspect_list = os.listdir(melspect_path)

mfcc_feat_path = os.path.join(slot_path, 'audio_features', 'mfcc_feat')
mfcc_feat_list = os.listdir(mfcc_feat_path)

img_label_df = pd.read_csv(img_label_dir, sep='\t')
audio_label_df = pd.read_csv(audio_label_dir, sep='\t')

# first logs
if img_label_df.shape[0] != audio_label_df.shape[0]:
    logging.error(f"\nin labels file the number of imgs is different from the one of audios \n"
                  f"there are {img_label_df.shape[0]} imgs and {audio_label_df.shape[0]} audios")
if len(img_list) != len(audio_list):
    logging.error(f"\nin folders the number of imgs is different from the one of audios \n"
                  f"there are {len(img_list)} imgs and {len(audio_list)} audios")

n_files = len(img_list)
empty_str_lst = ['']*n_files
empty_array = np.empty(n_files)
empty_array[:] = np.nan

addressees = ['NAO', 'PLEFT', 'PRIGHT', 'GROUP']
labels = [0, 1, 2, 3]
# zip method --> iterator of tuples --> dict method --> dictionary
labels_dict = dict(zip(addressees, labels))
print(labels_dict)

d = {'LABEL': empty_array, 'ADDRESSEE': empty_str_lst, 'SLOT': empty_str_lst, 'SPEAKER': empty_array, 'N_INTERVAL': empty_array,
     'START_MSEC': empty_array, 'END_MSEC': empty_array, 'FRAME_MSEC': empty_array, 'DURATION_MSEC': empty_array,
     'IMG': empty_str_lst, 'POSE_SPEAKER': empty_str_lst, 'FACE_SPEAKER': empty_str_lst, 'POSE_OTHER': empty_str_lst,
     'FACE_OTHER': empty_str_lst, 'AUDIO': empty_str_lst, 'MEL_SPECT': empty_str_lst, 'MFCC_FEAT': empty_str_lst}
all_labels_df = pd.DataFrame(data=d)

all_labels_df['SLOT'] = slot
all_labels_df['AUDIO'] = audio_label_df['FILE_NAME']

for i, audio in enumerate(all_labels_df['AUDIO']):

    index_file = audio_label_df.loc[audio_label_df['FILE_NAME'] == audio].index[0]
    slot = all_labels_df['SLOT'].loc[i]
    addressee = audio_label_df['LABEL_ADDRESSEE'].loc[index_file]
    speaker = audio_label_df['SPEAKER'].loc[index_file]
    n_interval = audio_label_df['N_INTERVAL'].loc[index_file]
    start_msec = audio_label_df['START_MSEC'].loc[index_file]
    duration_msec = audio_label_df['DURATION'].loc[index_file]
    frame_msec = int(start_msec + duration_msec/2)
    end_msec = start_msec + duration_msec

    img = f'slot{slot}_sp{speaker}_add{addressee}_video_frame_msec_{frame_msec}.jpg'
    pose_speaker = f'pose_speaker_slot{slot}_sp{speaker}_add{addressee}_video_frame_msec_{frame_msec}.npy'
    pose_other = f'pose_other_1_slot{slot}_sp{speaker}_add{addressee}_video_frame_msec_{frame_msec}.npy'
    face_speaker = f'face_speaker_slot{slot}_sp{speaker}_add{addressee}_video_frame_msec_{frame_msec}.jpg'
    face_other = f'face_other_1_slot{slot}_sp{speaker}_add{addressee}_video_frame_msec_{frame_msec}.jpg'
    mel_spect = f'melspec_slot{slot}_sp{speaker}_add{addressee}_audio_trimmed_msec_{start_msec}_{end_msec}.jpg'
    mfcc_feat = f'mfcc_feat_slot{slot}_sp{speaker}_add{addressee}_audio_trimmed_msec_{start_msec}_{end_msec}.npy'

    all_labels_df.loc[i, 'LABEL'] = labels_dict[addressee]
    all_labels_df.loc[i, 'ADDRESSEE'] = addressee
    all_labels_df.loc[i, 'SPEAKER'] = speaker
    all_labels_df.loc[i, 'N_INTERVAL'] = n_interval
    all_labels_df.loc[i, 'START_MSEC'] = start_msec
    all_labels_df.loc[i, 'END_MSEC'] = end_msec
    all_labels_df.loc[i, 'FRAME_MSEC'] = frame_msec
    all_labels_df.loc[i, 'DURATION_MSEC'] = duration_msec

    info_slot = audio.split('_')[0].split('slot')[-1]
    info_speaker = int(audio.split('_')[1].split('sp')[-1])
    info_addressee = audio.split('_')[2].split('add')[-1]
    info_start_msec = int(audio.split('_')[-2])
    info_end_msec = int(audio.split('_')[-1].split('.')[0])

    # second logs
    if slot != info_slot:
        logging.error(f"\nslot info in filename is different from the one in label file \n"
                      f"in filename it is {info_slot}, whereas in label file is {slot}")
    if addressee != info_addressee:
        logging.error(f"\naddressee info in filename is different from the one in label file \n"
                      f"in filename it is {info_addressee}, whereas in label file is {addressee}")
    if speaker != info_speaker:
        logging.error(f"\nspeaker info in filename is different from the one in label file \n"
                      f"in filename it is {info_speaker}, whereas in label file is {speaker}")
    if start_msec != info_start_msec:
        logging.error(f"\nstart_msec info in filename is different from the one in label file \n"
                      f"in filename it is {info_start_msec}, whereas in label file is {frame_msec}")

    # third logs
    if img in img_list:
        all_labels_df.loc[i, 'IMG'] = img
    else:
        logging.error(f"\nthere is a problem with the name of img, which should be:\n"
                      f"{img}")

    if pose_speaker in pose_speaker_list:
        all_labels_df.loc[i, 'POSE_SPEAKER'] = pose_speaker
    else:
        warnings.warn(f"\npose_speaker file not found in the correct folder\n"
                      f"maybe it is a problem with the name of pose_speaker, which should be:\n"
                      f"{pose_speaker}")

    if face_speaker in face_speaker_list:
        all_labels_df.loc[i, 'FACE_SPEAKER'] = face_speaker
    else:
        all_labels_df.loc[i, 'FACE_SPEAKER'] = "missing"
        face_speaker_not_detected += 1

    if pose_other in pose_other_list:
        all_labels_df.loc[i, 'POSE_OTHER'] = pose_other
    else:
        warnings.warn(f"\npose_other file not found in the correct folder\n"
                      f"maybe it is a problem with the name of pose_other, which should be:\n"
                      f"{pose_other}")

    if face_other in face_other_list:
        all_labels_df.loc[i, 'FACE_OTHER'] = face_other
    else:
        face_other_not_detected += 1
        all_labels_df.loc[i, 'FACE_OTHER'] = "missing"

    if mel_spect in melspect_list:
        all_labels_df.loc[i, 'MEL_SPECT'] = mel_spect
    else:
        logging.error(f"\nmel_spect file not found in the correct folder\n"
                      f"maybe it is a problem with the name of mel_spect, which should be:\n"
                      f"{mel_spect}")

    if mfcc_feat in mfcc_feat_list:
        all_labels_df.loc[i, 'MFCC_FEAT'] = mfcc_feat
    else:
        logging.error(f"\nthere is a problem with the name of mfcc_feat, which should be:\n"
                      f"{mfcc_feat}")


print(f'in total there are {face_speaker_not_detected} faces of the speaker not detected \n '
      f'and {face_other_not_detected} faces of the other not detected')


all_labels_df['LABEL'].astype(int)

print(all_labels_df)
final_csv = 'all_labels.csv'
save_dir = os.path.join(label_path, final_csv)

all_labels_df.to_csv(save_dir, sep='\t', index_label='index')

