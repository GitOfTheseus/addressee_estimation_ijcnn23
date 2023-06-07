import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io
from B_TRAIN.utils import translate_pose, calculate_possible_shift

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

