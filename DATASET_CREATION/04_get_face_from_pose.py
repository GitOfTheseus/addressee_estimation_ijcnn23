import numpy as np
import pandas as pd
import cv2
import os
import math

# define parameters for input
dataset_path = '/home/carlo/Documents/AddEstModel/dataset'
slot = '15'
pose_path = os.path.join(dataset_path, slot, "pose")
img_path = os.path.join(dataset_path, slot, "img")
face_path = os.path.join(dataset_path, slot, "face")
if not os.path.isdir(face_path):
    os.mkdir(face_path)

# let's downscale the image using new  width and height
new_width, new_height = 50, 50
new_size = (new_width, new_height)

def transformation_points_to_pixel(pose):
    #print(pose)
    for i, p in enumerate(pose):
        #print(i)
        #print(pose[i])
        # transformation from [-1,-1,1,1] to [0,0,2,2]
        pose[i][0] = pose[i][0] + 1
        pose[i][1] = pose[i][1] + 1
        #print(pose[i])
        # transformation from [0,0,2,2] to [0,0,img.height,img.width] to have coordinates in pixels
        pose[i][0] = int(pose[i][0] * img_width / 2)
        pose[i][1] = int(pose[i][1] * img_height / 2)
        #print(pose[i])

    return pose

def get_face_coordinates(pose, img_height, img_width):

    face_detected = False
    face_coordinates = [0,0,0,0]
    center_face = [int(pose[0][0]), int(pose[0][1])]
    base_neck = [int(pose[1][0]), int(pose[1][1])]
    left_ear = [int(pose[16][0]), int(pose[16][1])]
    right_ear = [int(pose[17][0]), int(pose[17][1])]

    # if face was detected
    if center_face[0] != 0:

        face_detected = True

        dist1 = int(math.dist(center_face, base_neck) * 3 / 5)
        dist2 = int(math.dist(center_face, left_ear))
        dist3 = int(math.dist(center_face, right_ear))

        if base_neck[0] == 0:
            dist1 = 0
        if left_ear[0] == 0:
            dist2 = 0
        if right_ear[0] == 0:
            dist3 = 0

        print(dist1, dist2, dist3)

        ray = max(dist1, dist2, dist3, 30)
        if ray == 0:
            face_detected = False

        over_edge = int(center_face[1] - ray)
        left_edge = center_face[0] - ray
        right_edge = center_face[0] + ray
        bottom_edge = center_face[1] + ray

        if over_edge < 0:
            bottom_edge -= over_edge
            over_edge = 0
        if left_edge < 0:
            right_edge -= left_edge
            left_edge = 0
        if bottom_edge > img_height:
            over_edge -= bottom_edge - img_height
            bottom_edge = img_height
        if right_edge > img_width:
            left_edge -= right_edge - img_width
            right_edge = img_width

        face_coordinates = [over_edge, bottom_edge, left_edge, right_edge]

    return face_detected, face_coordinates



for role in os.listdir(pose_path):

    #print(role)
    pose_role_path = os.path.join(pose_path, role)
    face_role_path = os.path.join(face_path, role)
    if not os.path.isdir(face_role_path):
        os.mkdir(face_role_path)

    list_poses = os.listdir(pose_role_path)
    #print(list_poses)
    for pose_file in list_poses:

        part_to_remove = pose_file.split('slot')[0]
        img_file = pose_file.replace(part_to_remove, "").replace(".npy", ".jpg")

        img_dir = os.path.join(img_path, img_file)
        img = cv2.imread(img_dir, cv2.IMREAD_COLOR)
        img_height, img_width = img.shape[0], img.shape[1]

        face_file = pose_file.replace(".npy", ".jpg").replace("pose", "face")
        if face_file == "face_other_1_slot12_sp1_addGROUP_video_frame_msec_706000.jpg":
            print(pose)
            print(face_coordinates, face_detected)
        face_dir = os.path.join(face_role_path, face_file)
        pose_dir = os.path.join(pose_role_path, pose_file)
        pose = np.load(pose_dir)
        #print(pose)
        pose = transformation_points_to_pixel(pose)
        face_detected, face_coordinates = get_face_coordinates(pose, img_height, img_width)

        if face_detected:

            crop = img[face_coordinates[0]:face_coordinates[1], face_coordinates[2]:face_coordinates[3]]
            resized = cv2.resize(crop, new_size, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(face_dir, resized)

            #cv2.imshow('original', img)
            #cv2.imshow('cropped', crop)
            #cv2.imshow('resized', resized)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()


## missing label file







