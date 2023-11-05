import cv2 as cv
import os
import numpy as np
import pandas as pd

dataset_path = '.../dataset_slots'
label_filename = 'lstm_label_no_doubles.csv'
slot_folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
slot_folders.sort()
print(slot_folders)
for slot in slot_folders:
    print(slot)

    slot_df = pd.read_csv(os.path.join(dataset_path, slot, label_filename), sep="\t", index_col='index')
    slot_folders.sort()

    new_df = slot_df[(slot_df["LABEL"] == 1) | (slot_df["LABEL"] == 2)].copy(deep=True)

    last_sequence = slot_df['SEQUENCE'].loc[slot_df.index[-1]]
    n = 0
    n_sequence = last_sequence + 1

    for i in list(new_df.index.values):

        if (slot_df.loc[i, 'ADDRESSEE'] == 'PLEFT' or slot_df.loc[i, 'ADDRESSEE'] == 'PRIGHT'):
            if slot_df.loc[i, 'ADDRESSEE'] == 'PLEFT':
                new_addressee = 'PRIGHT'
                new_label = 2
            elif slot_df.loc[i, 'ADDRESSEE'] == 'PRIGHT':
                new_addressee = 'PLEFT'
                new_label = 1

            new_df.loc[i, 'ADDRESSEE'] = new_addressee
            new_df.loc[i, 'LABEL'] = new_label
            new_df.loc[i, 'SEQUENCE'] = n_sequence

            n += 1
            if n == 10:
                n = 0
                n_sequence += 1

            # face speaker
            face_s_file = os.path.join(dataset_path, slot, 'face_s', new_df.loc[i, 'FACE_SPEAKER'])
            img = cv.imread(face_s_file)
            img_flip_lr = cv.flip(img, 1)
            new_img_file = new_df.loc[i, 'FACE_SPEAKER'].replace('.jpg', '_flipped') + '.jpg'
            new_df.loc[i, 'FACE_SPEAKER'] = new_img_file
            cv.imwrite(os.path.join(dataset_path, slot, 'face_s', new_img_file), img_flip_lr)

            # face other
            face_o_file = os.path.join(dataset_path, slot, 'face_o', new_df.loc[i, 'FACE_OTHER'])
            img = cv.imread(face_o_file)
            img_flip_lr = cv.flip(img, 1)
            new_img_file = new_df.loc[i, 'FACE_OTHER'].replace('.jpg', '_flipped') + '.jpg'
            new_df.loc[i, 'FACE_OTHER'] = new_img_file
            cv.imwrite(os.path.join(dataset_path, slot, 'face_o', new_img_file), img_flip_lr)

            # pose speaker
            pose_s_file = os.path.join(dataset_path, slot, 'pose_s', new_df.loc[i, 'POSE_SPEAKER'])
            pose = np.load(pose_s_file)
            for k, p in enumerate(pose):
                if p[2] != 0:
                    p[0] = p[0] * -1
                pose[k,:] = p
            new_pose_file = new_df.loc[i, 'POSE_SPEAKER'].replace('.npy', '_flipped') + '.npy'
            new_df.loc[i, 'POSE_SPEAKER'] = new_pose_file
            np.save(os.path.join(dataset_path, slot, 'pose_s', new_pose_file), pose)

            pose_o_file = os.path.join(dataset_path, slot, 'pose_o', new_df.loc[i, 'POSE_OTHER'])
            pose = np.load(pose_o_file)
            for k, p in enumerate(pose):
                if p[2] != 0:
                    p[0] = p[0] * -1
                pose[k,:] = p
            new_pose_file = new_df.loc[i, 'POSE_OTHER'].replace('.npy', '_flipped') + '.npy'
            new_df.loc[i, 'POSE_OTHER'] = new_pose_file
            np.save(os.path.join(dataset_path, slot, 'pose_o', new_pose_file), pose)

            mel_file = os.path.join(dataset_path, slot, 'mel', new_df.loc[i, 'MEL_SPECT'])
            img = cv.imread(mel_file)
            new_img_file = new_df.loc[i, 'MEL_SPECT'].replace('.jpg', '_flipped') + '.jpg'
            new_df.loc[i, 'MEL_SPECT'] = new_img_file
            cv.imwrite(os.path.join(dataset_path, slot, 'mel', new_img_file), img)

            mfcc_file = os.path.join(dataset_path, slot, 'mfcc', new_df.loc[i, 'MFCC_FEAT'])
            features = np.load(mfcc_file)
            new_mfcc_file = new_df.loc[i, 'MFCC_FEAT'].replace('.npy', '_flipped') + '.npy'
            new_df.loc[i, 'MFCC_FEAT'] = new_mfcc_file
            np.save(os.path.join(dataset_path, slot, 'mfcc', new_mfcc_file), features)

    slot_df = pd.concat([slot_df, new_df], ignore_index=True)
    slot_df.to_csv(os.path.join(dataset_path, slot, 'lstm_label_augmented_no_doubles.csv'), sep='\t', index_label='index')
