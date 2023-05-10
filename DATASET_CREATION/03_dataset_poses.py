import sys
import cv2
import os
from sys import platform
import argparse
import time
import glob
import json
import numpy as np
import pandas as pd
import warnings

dataset_path = '/home/carlo/Documents/AddEstModel/dataset'
slot = '27'
img_path = os.path.join(dataset_path, slot, 'img')
csv_file = 'labels_video_slot' + slot + '.csv'
csv_dir = os.path.join(dataset_path, slot, 'labels', csv_file) # withindex_spe info temporally aligned

img_label = pd.read_csv(csv_dir, sep='\t', index_col=0)
lst = ['']*img_label.shape[0]
img_label['POSE_OTHER_FILE'] = lst
img_label['POSE_SPEAKER_FILE'] = lst
print(img_label['SPEAKER'])

save_pose_path = os.path.join(dataset_path, slot, 'pose')
save_speaker_path = os.path.join(save_pose_path, 'speaker')
save_other_path = os.path.join(save_pose_path, 'other')
if not os.path.exists(save_speaker_path):
    os.makedirs(save_speaker_path)
if not os.path.exists(save_other_path):
    os.makedirs(save_other_path)

try:
    # Import Openpose (Ubuntu)
    os.path.dirname(os.path.realpath(__file__))

    print(os.path.realpath(__file__))
    sys.path.append('../../openpose/build/python')
    print('current directory is ', os.getcwd())

    from openpose import pyopenpose as op

except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e



def spatial_ordering(datum):
    print("# people detected: ", str(len(datum.poseKeypoints)))
    core_list = np.zeros((len(datum.poseKeypoints),), dtype=[('x', 'float32'), ('y', 'float32')])
    new_core_list = np.zeros((2,), dtype=[('x', 'float32'), ('y', 'float32')])
    probabilities = []
    for n_body in range(len(datum.poseKeypoints)):
        body = datum.poseKeypoints[n_body]
        x = []
        y = []
        z = []

        for joints in body:
            if joints[2] != 0:
                x.append(joints[0])
                y.append(joints[1])
                z.append(joints[2])
        x_core = np.mean(np.array(x))
        y_core = np.mean(np.array(y))
        if len(datum.poseKeypoints) > 2:
            #print(datum.poseKeypoints)
            z_probability = np.mean(np.array(z))
            probabilities.append(z_probability)

        core_list['x'][n_body] = x_core
        core_list['y'][n_body] = y_core

    poses = []

    # index_sorted contains indexes of body-postures sorted from left to right
    index_sorted = np.argsort(core_list, order=('x', 'y'))
    print("index_sorted is ", index_sorted)

    if len(datum.poseKeypoints) > 2:

        max1 = max(probabilities)
        new_prob = []
        for n in probabilities:
            if n != max1:
                new_prob.append(n)
        max2 = max(new_prob)
        new_prob.remove(max2)
        print(max1, max2)
        print(probabilities)
        print(new_prob)
        index_ok = [probabilities.index(max1), probabilities.index(max2)]
        print(index_ok)
        new_index_sorted = []
        for i in index_sorted:
            if i in index_ok:
                new_index_sorted.append(i)
        for i in index_sorted:
            if i not in index_ok:
                new_index_sorted.append(i)
        index_sorted = new_index_sorted
        print("new index_sorted is ", index_sorted)

    return index_sorted

def check_speaker_body(index_sorted, speaker_id):

    # in Vernissage dataset there are only 2 people:  #1 left #2 right

    if speaker_id == 1:
        index_speaker = index_sorted[0]
    elif speaker_id == 2:
        index_speaker = index_sorted[1]
    else:
        print("NO SPEAKER! check loaded files")

    return index_speaker

def save_pose(datum, index_sorted, index_speaker, file_npy, save_speaker_path, save_other_path, img_label):
    print("\nprinting and poses from left to right (robot's egocentric perspective)")

    n_others = 0
    try:
        for index in index_sorted:

            if index == index_speaker:
                file_name = 'pose_speaker_' + file_npy
                save_path = save_speaker_path
                img_label['POSE_SPEAKER_FILE'].at[index_img] = file_name
            else:
                n_others += 1
                file_name = f"pose_other_{str(n_others)}_{file_npy}"
                save_path = save_other_path
                img_label['POSE_OTHER_FILE'].at[index_img] = file_name

            pose = np.asarray(datum.poseKeypoints[index])
            #print("\nPOSE #", str(index), "\n", str(pose))
            save_dir = os.path.join(save_path, file_name)
            np.save(save_dir, pose)
    except Warning:
        print("not possible to get poses for this image")

        #with open(file_path, 'w+') as f:
        #    json.dump(pose.tolist(), f)

    return img_label

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", default=img_path,
                    help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
# params["model_folder"] = "../../../models/"
params["model_folder"] = "/home/carlo/Documents/openpose/models"
params["keypoint_scale"] = 4  # To rescale bodily keypoints
params["model_pose"] = "COCO"  # To manage the model that computes keypoints
print("the MODEL chosen for pose detection is ", params["model_pose"])

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1]) - 1:
        next_item = args[1][i + 1]
    else:
        next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-', '')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-', '')
        if key not in params: params[key] = next_item

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

os.chdir(img_path)
imagePaths = op.get_images_on_directory(img_path)
print(imagePaths)
all_poses = os.listdir(save_speaker_path)
for i, image in enumerate(imagePaths):
    print(i, imagePaths[i].split('/')[-1])
    file_npy = image.split('/')[-1].replace('.jpg', '.npy')
    name_check = 'pose_speaker_' + file_npy
    if name_check not in all_poses:
        datum = op.Datum()
        # read image
        imageToProcess = cv2.imread(image)
        datum.cvInputData = imageToProcess

        # detect body keypoints
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        index_sorted = spatial_ordering(datum)
        index_img = img_label.loc[img_label['IMG_NAME']==image.split('/')[-1]].index[0]

        speaker = img_label.loc[index_img]['SPEAKER']
        print(f"at index {index_img} got speaker {speaker}")
        index_speaker = check_speaker_body(index_sorted, speaker_id=speaker)
        print('index_speaker is ', index_speaker)

        img_label = save_pose(datum, index_sorted, index_speaker, file_npy, save_speaker_path, save_other_path, img_label)


new_label_file = 'labels_poses_' + slot + '.csv'
new_csv_dir = os.path.join(dataset_path, slot, 'labels', new_label_file) # with info temporally aligned
img_label.to_csv(new_csv_dir, sep='\t')