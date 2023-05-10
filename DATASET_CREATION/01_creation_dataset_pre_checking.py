from moviepy.editor import VideoFileClip, AudioFileClip
import numpy as np
import pandas as pd
import os


### define parameters and paths for input

# general path of the dataset
dataset_path = '/home/carlo/Documents/AddEstModel/dataset'
# n. slot of the dataset
slot = '27'
# person id to consider as speaker: 1 is the person on the left, 2 the one on the right
person = 2
# duration of the interval in msecs to trim audio and to take frame of the video: save data each x msecs
saving_interval = 80
# list with the possible addresses
addressees = ['NAO', 'GROUP', 'PRIGHT', 'PLEFT']
# csv file with the temporal info and labels needed to save our dataset
csv_file = 'dataset_person' + str(person) + '.csv'
csv_dir = os.path.join(dataset_path, slot, csv_file)  # with info temporally aligned
df = pd.read_csv(csv_dir, sep='\t')     # read temporally aligned information
# read video merged
original_videomerged_file = 'naoav-merged.ogv'
original_videomerged_dir = os.path.join(dataset_path, slot, original_videomerged_file)
original_videomerged_clip = VideoFileClip(original_videomerged_dir)
# read video nao
original_video_file = 'naovideo.avi'
original_video_dir = os.path.join(dataset_path, slot, original_video_file)
original_video_clip = VideoFileClip(original_video_dir)
# read audio nao and save it with 48000 fps
audio_file = 'naoaudio.wav'
audio48K_file = "naoaudio48K.wav"
audio_dir = os.path.join(dataset_path, slot, audio_file)
original_audioclip = AudioFileClip(audio_dir, fps=48000)
print(original_audioclip.fps, original_audioclip.duration, original_audioclip.start)
all_file_slot = os.listdir(os.path.join(dataset_path, slot))
if audio48K_file not in all_file_slot:
    original_audioclip.write_audiofile(os.path.join(dataset_path, slot, audio48K_file), ffmpeg_params=["-ac", "1"])

# name of the file to save data
file = 'slot' + slot + '_sp' + str(person)
# path to save data
check_path = os.path.join(dataset_path, slot, 'to_check')
if not os.path.isdir(check_path):
    os.mkdir(check_path)


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
        #video_tstamp_start += 1650000
        print("video timestamp start at: ", video_tstamp_start)
    # time difference between original video and audio in msecs
    audio_diff_ms = round((video_tstamp_start - audio_tstamp_start) / 1000)
    print("audio time difference in ms is ", audio_diff_ms)
    # time difference between original video and merged video in msecs
    video_merged_diff_ms = round((video_tstamp_start - videomerged_tstamp_start) / 1000)
    print("video merged time difference in ms is ", video_merged_diff_ms)

    return audio_diff_ms, video_merged_diff_ms

# function to take start and stop of each speech interval and save them on a dataframe
def take_speech_intervals(addressee):
    df_speech = df[df['ADDRESSEE'] == addressee]
    video_starts = []
    video_stops = []
    audio_starts = []
    audio_stop = []
    for k, f in enumerate(df_speech['MILLISECONDS'].index):
        if f == df_speech.index[0]:
            video_starts.append(df_speech['MILLISECONDS'].at[f])
            audio_starts.append(df_speech['AUDIO_TIME_(mus)'].at[f])
        elif f == df_speech.index[-1]:
            video_stops.append(df_speech['MILLISECONDS'].at[f] + 40)
            audio_stop.append(df_speech['AUDIO_TIME_(mus)'].at[f] + 40000)
        elif df_speech['MILLISECONDS'].at[f] != df_speech['MILLISECONDS'].at[df_speech.index[k - 1]] + 40:
            video_stops.append(df_speech['MILLISECONDS'].at[df_speech.index[k - 1]] + 40)  # add 40 ms
            video_starts.append(df_speech['MILLISECONDS'].at[f])
            audio_stop.append(df_speech['AUDIO_TIME_(mus)'].at[df_speech.index[k - 1]] + 40000)  # add 40 ms
            audio_starts.append(df_speech['AUDIO_TIME_(mus)'].at[f])

    d_video = {'STARTS': video_starts, 'STOPS': video_stops}
    d_audio = {'STARTS': audio_starts, 'STOPS': audio_stop}
    video_int = pd.DataFrame(data=d_video)
    audio_int = pd.DataFrame(data=d_audio)

    return video_int, audio_int


### core of the code

# calculate the time_difference through timestamp information for alignment
audio_diff_ms, video_merged_diff_ms = time_difference(dataset_path, slot)

for addressee in addressees:

        filename = file + '_add' + addressee + '_'
        # take all the start/end intervals for each addressee
        video_int, audio_int = take_speech_intervals(addressee)
        # for each speech interval
        for i in video_int.index:

            # get start and end of the intervals
            start_ms = int(video_int['STARTS'].at[i])
            end_ms = int(video_int['STOPS'].at[i])
            print("interval is ", end_ms-start_ms)

            # do not consider intervals shorter than saving interval
            if end_ms-start_ms >= saving_interval:

                # round intervals to multiple of saving intervals (increase ratio)
                if (end_ms-start_ms) % saving_interval != 0:
                    time_2get_round = saving_interval - ((end_ms-start_ms) % saving_interval)
                    print("interval was ", end_ms - start_ms)
                    end_ms = int(video_int['STOPS'].at[i] + time_2get_round)
                    print("now interval is ", end_ms - start_ms)
                    if (end_ms - start_ms) % saving_interval == 0:
                        print("which is multiple of ", saving_interval)

                # time alignment
                start_merged = start_ms + video_merged_diff_ms
                end_merged = end_ms + video_merged_diff_ms
                start_audio = start_ms + audio_diff_ms
                end_audio = end_ms + audio_diff_ms
                # get audio in sec
                start_audio_s = start_audio / 1000
                end_audio_s = end_audio / 1000

                # trim videomerged to check alignment and labelization
                videomerged_clip1 = original_videomerged_clip.subclip(start_merged / 1000, end_merged / 1000)
                videomerged_file = filename + f"videomerged_msec_{start_ms}_{end_ms}.mp4"
                videomerged_dir = os.path.join(check_path, videomerged_file)
                videomerged_clip1.write_videofile(videomerged_dir)

                # trim audio with moviepy to check alignment
                audioclip1 = original_audioclip.subclip(start_audio_s, end_audio_s)
                audio_file = filename + f"audio_msec_{start_ms}_{end_ms}.wav"
                audio_dir = os.path.join(check_path, audio_file)
                audioclip1.write_audiofile(audio_dir)

                # trim video with moviepy to check alignment
                video_clip1 = original_video_clip.subclip(start_ms / 1000, end_ms / 1000)
                video_file = filename + f"video_msec_{start_ms}_{end_ms}.mp4"
                video_dir = os.path.join(check_path, video_file)
                video_clip1.write_videofile(video_dir)
