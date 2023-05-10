import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from skimage import io
import os
from utils import modify_df_for_audio, translate_pose, calculate_possible_shift

torch.manual_seed(0)
np.random.seed(0)

# Need to override __init__, __len__, __getitem__
# as per datasets requirement
class CustomDataset_vision(torch.utils.data.Dataset):

    def __init__(self, label_dir, root_dir, transform_face, transform_pose, phase):

        self.data = pd.read_csv(label_dir, sep='\t', index_col=False, dtype=str)
        self.root_dir = root_dir
        self.transform_face = transform_face
        self.transform_pose = transform_pose
        self.sourceTransform = None
        self.last_sequence = 0
        self.last_shift = 0
        self.left_max_shift, self.right_max_shift = 0, 0
        self.phase = phase
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        slot = self.data['SLOT'][idx]
        if pd.isna(slot):
            slot = ""
        if slot == '9':
            slot = '09'

        # AGGIUNTO
        sequence = self.data['SEQUENCE'][idx]
        interval = self.data['N_INTERVAL'][idx]
        face_dir = os.path.join(self.root_dir, slot, 'face_s', self.data['FACE_SPEAKER'][idx])
        face_dir = face_dir.replace('npy', 'jpg')
        face = io.imread(face_dir)

        pose_dir = os.path.join(self.root_dir, slot, 'pose_s', self.data['POSE_SPEAKER'][idx])
        pose = np.load(pose_dir)

        if sequence != self.last_sequence:
            do_shift = True
            pose_path = os.path.join(self.root_dir, slot, 'pose_s')
            self.left_max_shift, self.right_max_shift = calculate_possible_shift(self.data, pose_path, sequence)
        else:
            do_shift = False
        self.last_sequence = sequence

        if self.phase != 'test':
            pose, self.last_shift = translate_pose(pose, do_shift, self.left_max_shift, self.right_max_shift, self.last_shift)

        label = int(float(self.data['LABEL'][idx]))

        if self.transform_face:
            face = self.transform_face(face)

        if self.transform_pose:
            pose = self.transform_pose(pose)

        sample = {'face': face, 'pose': pose, 'label': label}

        if self.sourceTransform:
            sample = self.sourceTransform(sample)
        # AGGIUNTO SEQUENCE
        return face, pose, label, sequence, interval


class CustomDataset_pose(torch.utils.data.Dataset):

    def __init__(self, label_dir, root_dir, transform_pose, phase):

        self.data = pd.read_csv(label_dir, sep='\t', index_col=False, dtype=str)
        self.root_dir = root_dir
        self.transform_pose = transform_pose
        self.sourceTransform = None
        self.last_sequence = 0
        self.last_shift = 0
        self.left_max_shift, self.right_max_shift = 0, 0
        self.phase = phase
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        slot = self.data['SLOT'][idx]
        if slot == '9':
            slot = '09'
        if pd.isna(slot):
            slot = ""

        sequence = self.data['SEQUENCE'][idx]
        interval = self.data['N_INTERVAL'][idx]

        pose_dir = os.path.join(self.root_dir, slot, 'pose_s', self.data['POSE_SPEAKER'][idx])
        pose = np.load(pose_dir)

        if sequence != self.last_sequence:
            do_shift = True
            pose_path = os.path.join(self.root_dir, slot, 'pose_s')
            self.left_max_shift, self.right_max_shift = calculate_possible_shift(self.data, pose_path, sequence)
        else:
            do_shift = False
        self.last_sequence = sequence

        if self.phase != 'test':
            pose, self.last_shift = translate_pose(pose, do_shift, self.left_max_shift, self.right_max_shift, self.last_shift)

        label = int(float(self.data['LABEL'][idx]))

        if self.transform_pose:
            #pose = np.concatenate((pose, pose, pose, pose, pose, pose))
            pose = self.transform_pose(pose)

        sample = {'pose': pose, 'label': label}

        if self.sourceTransform:
            sample = self.sourceTransform(sample)

        return pose, label, sequence, interval


class CustomDataset_vision_with_other(torch.utils.data.Dataset):

    def __init__(self, label_dir, root_dir, transform_face, transform_pose):

        self.data = pd.read_csv(label_dir, sep='\t', index_col=False, dtype=str)
        self.root_dir = root_dir
        self.transform_face = transform_face
        self.transform_pose = transform_pose
        self.sourceTransform = None
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        slot = self.data['SLOT'][idx]
        if slot == '9':
            slot = '09'

        face_speaker_dir = os.path.join(self.root_dir, slot, 'face_s', self.data['FACE_SPEAKER'][idx])
        face_speaker = io.imread(face_speaker_dir)

        face_other_dir = os.path.join(self.root_dir, slot, 'face_o', self.data['FACE_OTHER'][idx])
        face_other = io.imread(face_other_dir)

        pose_speaker_dir = os.path.join(self.root_dir, slot, 'pose_s', self.data['POSE_SPEAKER'][idx])
        pose_speaker = np.load(pose_speaker_dir)

        pose_other_dir = os.path.join(self.root_dir, slot, 'pose_o', self.data['POSE_OTHER'][idx])
        pose_other = np.load(pose_other_dir)

        label = int(float(self.data['LABEL'][idx]))

        if self.transform_face:
            face_speaker = self.transform_face(face_speaker)
            face_other = self.transform_face(face_other)

        if self.transform_pose:
            pose_speaker = self.transform_pose(pose_speaker)
            pose_other = self.transform_pose(pose_other)

        sample = {'face_speaker': face_speaker, 'pose_speaker': pose_speaker,
                  'face_other': face_other, 'pose_other': pose_other, 'label': label}

        if self.sourceTransform:
            sample = self.sourceTransform(sample)

        return face_speaker, pose_speaker, face_other, pose_other, label

class CustomDataset_vision_audio(torch.utils.data.Dataset):

    def __init__(self, label_dir, root_dir, transform_face, transform_pose, transform_spect):

        self.data = pd.read_csv(label_dir, sep='\t', index_col=False, dtype=str)
        self.root_dir = root_dir
        self.transform_face = transform_face
        self.transform_pose = transform_pose
        self.transform_spect = transform_spect
        self.sourceTransform = None
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        slot = self.data['SLOT'][idx]
        if slot == '9':
            slot = '09'

        # AGGIUNTO
        sequence = self.data['SEQUENCE'][idx]
        interval = self.data['N_INTERVAL'][idx]
        face_dir = os.path.join(self.root_dir, slot, 'face_s', self.data['FACE_SPEAKER'][idx])
        face = io.imread(face_dir)

        pose_dir = os.path.join(self.root_dir, slot, 'pose_s', self.data['POSE_SPEAKER'][idx])
        pose = np.load(pose_dir)

        spect_dir = os.path.join(self.root_dir, slot, 'mel_cleaned', self.data['MEL_SPECT'][idx])
        if "flipped" in spect_dir:
            spect_dir = spect_dir.replace("_flipped", "")
        spect = io.imread(spect_dir)

        label = int(float(self.data['LABEL'][idx]))

        if self.transform_face:
            face = self.transform_face(face)

        if self.transform_pose:
            pose = self.transform_pose(pose)

        if self.transform_spect:
            spect = self.transform_spect(spect)

        sample = {'face': face, 'pose': pose, 'spect': spect, 'label': label}

        if self.sourceTransform:
            sample = self.sourceTransform(sample)
        # AGGIUNTO SEQUENCE
        return face, pose, spect, label, sequence, interval


class CustomDataset_audio(torch.utils.data.Dataset):

    def __init__(self, label_dir, root_dir, transform_spect):

        self.data = modify_df_for_audio(pd.read_csv(label_dir, sep='\t', index_col=False, dtype=str))
        self.root_dir = root_dir
        self.transform_spect = transform_spect
        self.sourceTransform = None

        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        slot = self.data['SLOT'][idx]
        if slot == '9':
            slot = '09'

        spect_dir = os.path.join(self.root_dir, slot, 'mel_cleaned', self.data['MEL_SPECT'][idx])
        if "flipped" in spect_dir:
            spect_dir = spect_dir.replace("_flipped", "")
        #print(spect_dir)
        spect = io.imread(spect_dir)

        label = int(float(self.data['LABEL'][idx]))
        if label >= 1:
            label = 1

        sequence = self.data['SEQUENCE'][idx]
        interval = self.data['N_INTERVAL'][idx]
        if self.transform_spect:
            spect = self.transform_spect(spect)

        sample = {'spect': spect, 'label': label}

        if self.sourceTransform:
            sample = self.sourceTransform(sample)

        return spect, label, sequence, interval

class CustomDataset_audio_cnn(torch.utils.data.Dataset):

    def __init__(self, label_dir, root_dir, transform_spect):

        self.data = pd.read_csv(label_dir, sep='\t', index_col=False, dtype=str)
        self.root_dir = root_dir
        self.transform_spect = transform_spect
        self.sourceTransform = None
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        slot = self.data['SLOT'][idx]
        if slot == '9':
            slot = '09'

        spect_dir = os.path.join(self.root_dir, slot, 'audio_sequences', 'mel_cnn', self.data['MEL_SPECT'][idx])
        spect = io.imread(spect_dir)

        label = int(float(self.data['LABEL2_INVOLVED'][idx]))

        if self.transform_spect:
            spect = self.transform_spect(spect)

        sample = {'spect': spect, 'label': label}

        if self.sourceTransform:
            sample = self.sourceTransform(sample)

        return spect, label
