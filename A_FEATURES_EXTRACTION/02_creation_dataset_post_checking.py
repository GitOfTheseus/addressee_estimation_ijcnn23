from moviepy.editor import VideoFileClip, AudioFileClip
from scipy.io import wavfile
import numpy as np
import pandas as pd
import os
import cv2

### define parameters and paths for input


# general path of the dataset
dataset_path = ''
# n. slot of the dataset
slot = '09'
# duration of the interval in msecs to trim audio and to take frame of the video: save data each x msecs
saving_interval = 80
# path where checked video interval are located
checked_path = os.path.join(dataset_path, slot, "checked")

original_video_file = 'naovideo.avi'
original_video_dir = os.path.join(dataset_path, slot, original_video_file)
original_video_clip = VideoFileClip(original_video_dir)
# read video file in frames with opencv
video_capture = cv2.VideoCapture(original_video_dir)
# read audio file saved in 48K fps
audio_file = 'naoaudio48K.wav'
audio_dir = os.path.join(dataset_path, slot, audio_file)
fs_wav, data_wav = wavfile.read(audio_dir)
print(fs_wav)
print(len(data_wav))
# normalize audio signal
data_wav_norm = data_wav / (2 ** 15)
print('Signal Duration = {} msecs'.
      format(data_wav.shape[0] / fs_wav * 1000))

# path to save video frames
save_img_path = os.path.join(dataset_path, slot, 'img')
if not os.path.isdir(save_img_path):
    os.mkdir(save_img_path)
# path to save trimmed audio file
save_audio_path = os.path.join(dataset_path, slot, 'audio')
if not os.path.isdir(save_audio_path):
    os.mkdir(save_audio_path)
# path to save csv file for labelling
labels_path = os.path.join(dataset_path, slot, 'labels')
if not os.path.exists(labels_path):
    os.makedirs(labels_path)
# files to save dataframes with labels
labels_video_csv = 'labels_video_' + 'slot' + slot + '.csv'
labels_video_csv_dir = os.path.join(labels_path, labels_video_csv)
labels_audio_csv = 'labels_audio_' + 'slot' + slot + '.csv'
labels_audio_csv_dir = os.path.join(labels_path, labels_audio_csv)
df_labels_video = pd.DataFrame()
df_labels_audio = pd.DataFrame()

SAVING_FRAMES_PER_SECOND = 25
# if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
saving_frames_per_second = min(original_video_clip.fps, SAVING_FRAMES_PER_SECOND)
# if SAVING_FRAMES_PER_SECOND is set to 0, step is 1/fps, else 1/SAVING_FRAMES_PER_SECOND
step = 1 / original_video_clip.fps if saving_frames_per_second == 0 else 1 / saving_frames_per_second
step_ms = step * 1000
print('step is', step)
int_index = 0
df_index = 0


# function to calculate time difference through timestamp information
def time_difference(dataset_path, slot):
    audio_tstamp_file = 'timestamp-naoaudio-begin.txt'
    videomerged_tstamp_file = 'timestamp-naoav-merged-begin.txt'
    video_tstamp_file = 'naovideo-stats.txt'

    audio_tstamp_dir = os.path.join(dataset_path, slot, audio_tstamp_file)
    videomerged_tstamp_dir = os.path.join(dataset_path, slot, videomerged_tstamp_file)
    video_tstamp_dir = os.path.join(dataset_path, slot, video_tstamp_file)

    with open(audio_tstamp_dir) as f:
        lines = f.readlines()
        audio_tstamp_start = int(lines[0])
        print("audio timestamp start at: ", audio_tstamp_start)

    with open(videomerged_tstamp_dir) as f:
        lines = f.readlines()
        videomerged_tstamp_start = int(lines[0])
        print("video merged timestamp start at: ", videomerged_tstamp_start)

    with open(video_tstamp_dir) as f:
        lines = f.readlines()
        index1 = lines[1].index(': ')
        index2 = lines[1].index(' mus')
        video_tstamp_start = int(lines[1][index1 + 2:index2])
        print("video timestamp start at: ", video_tstamp_start)

    audio_diff_ms = round((video_tstamp_start - audio_tstamp_start) / 1000)
    print("audio time difference in ms is ", audio_diff_ms)
    video_merged_diff_ms = round((video_tstamp_start - videomerged_tstamp_start) / 1000)
    print("video merged time difference in ms is ", video_merged_diff_ms)

    return audio_diff_ms, video_merged_diff_ms

# functions to save frame from video
def save_frame(video_capture, current_duration, filename, save_img_path):

    # name of the frame file
    frame_filename = filename + f"video_frame_msec_{int(current_duration)}.jpg"
    # specific directory of the file
    frame_dir = os.path.join(save_img_path, frame_filename)
    # set the video frame at the specific duration given by current_duration (calculated as the average time between
    # the start and the end of the interval saving_interval
    video_capture.set(cv2.CAP_PROP_POS_MSEC, current_duration)
    # get the frame at the wanted duration
    res, frame = video_capture.read()
    # save the frame
    #cv2.imwrite(frame_dir, frame)

    return frame_filename


### core of the code

# calculate the time difference between files
audio_diff_ms, video_merged_diff_ms = time_difference(dataset_path, slot)
# take the list of the checked file to take as intervals
checked_list = os.listdir(checked_path)
print(checked_list)

# for each checked interval take the frame and trim the audio of length saving_interval
for filename in checked_list:
    # take information from the name of the file
    file_info = filename.replace(".mp4", "").split("_")
    person = file_info[1][-1]
    addressee = file_info[2].replace('add', "")
    start_ms = int(file_info[5])
    end_ms = int(file_info[6])
    # count the n. of the speech interval
    int_index += 1
    new_name = f"slot{slot}_sp{person}_add{addressee}_"
    print(f"{filename} : interval n. {int_index}")

    # divide the speech interval in mini-intervals of length saving_interval
    for x in np.arange(start_ms, end_ms, saving_interval):
        # take the start and the end to trim the audio and to calculate the framing time
        start_int_ms = int(round(x))
        end_int_ms = int(round(start_int_ms + saving_interval))
        # take the mean as the moment when to take the frame from the video
        frame_ms = ((start_int_ms + end_int_ms) / 2)
        # save the frame
        frame_filename = save_frame(video_capture, frame_ms, new_name, save_img_path)
        # put information for labelling in a dataframe, then concatenate in another
        frame = {}
        df_index += 1
        frame['IMG_NAME'] = frame_filename
        frame['SPEAKER'] = person
        frame['LABEL_ADDRESSEE'] = addressee
        frame['START_MSEC'] = start_int_ms
        frame['DURATION'] = saving_interval
        frame['N_INTERVAL'] = int_index

        df_frame = pd.DataFrame(frame, index=[df_index])
        df_labels_video = pd.concat([df_labels_video, df_frame])

        # align time of the audio
        start_trim_ms = start_int_ms + audio_diff_ms
        end_trim_ms = end_int_ms + audio_diff_ms
        # calculate start and end of the mini-interval in frames
        start_trim_fs = start_trim_ms * fs_wav / 1000
        end_trim_fs = end_trim_ms * fs_wav / 1000
        # trim the signal
        segment = np.array(data_wav_norm[int(start_trim_fs):int(end_trim_fs)])
        # name of the audio trimmed file
        audio_file = new_name + f"audio_trimmed_msec_{start_int_ms}_{end_int_ms}.wav"
        audio_dir = os.path.join(save_audio_path, audio_file)
        # save the audio trimmed file
        #wavfile.write(audio_dir, fs_wav, segment)
        # put information for labelling in a dataframe, then concatenate in another
        audio_trimmed = {}
        audio_trimmed['FILE_NAME'] = audio_file
        audio_trimmed['SPEAKER'] = person
        audio_trimmed['LABEL_ADDRESSEE'] = addressee
        audio_trimmed['START_MSEC'] = start_int_ms
        audio_trimmed['DURATION'] = saving_interval
        audio_trimmed['N_INTERVAL'] = int_index

        df_audio = pd.DataFrame(audio_trimmed, index=[df_index])
        df_labels_audio = pd.concat([df_labels_audio, df_audio])

### saving dataframe of labels
df_labels_video.to_csv(labels_video_csv_dir, sep='\t')
df_labels_audio.to_csv(labels_audio_csv_dir, sep='\t')



