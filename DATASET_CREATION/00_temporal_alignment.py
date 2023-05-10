import pandas as pd
import xml.etree.ElementTree as ET
import os

### paths and parameters in input ###

# general path of the dataset
dataset_path = '/home/carlo/Documents/AddEstModel/dataset'
# n. slot of the dataset
slot = '30'
# person id to consider as speaker: 1 is the person on the left, 2 the one on the right
person = 2
# file with all the annotations for the specific slot and the specific person considered as speaker
xml_file = 'annotations_person' + str(person) + '.xml'
# file where needed info will be saved
csv_file = 'dataset_person' + str(person) + '.csv'
# file with timestamp info of the naoaudio.wav
audio_tstamp_file = 'timestamp-naoaudio-begin.txt'
# file with timestamp info of the naoav_merged.ogv
videomerged_tstamp_file = 'timestamp-naoav-merged-begin.txt'
# file with timestamp info of the naovideo.avi
video_tstamp_file = 'naovideo-stats.txt'
# directories where all these file are loaded or saved
xml_dir = os.path.join(dataset_path, slot, xml_file)
csv_dir = os.path.join(dataset_path, slot, csv_file)
audio_tstamp_dir = os.path.join(dataset_path, slot, audio_tstamp_file)
videomerged_tstamp_dir = os.path.join(dataset_path, slot, videomerged_tstamp_file)
video_tstamp_dir = os.path.join(dataset_path, slot, video_tstamp_file)

df = pd.DataFrame()
# get the information three to read info from xml file
tree = ET.parse(xml_dir)
root = tree.getroot()

# read timestamp audio
with open(audio_tstamp_dir) as f:
    lines = f.readlines()
    audio_tstamp_start = int(lines[0])
    print("audio timestamp start at: ", audio_tstamp_start)
# read timestamp video merged
with open(videomerged_tstamp_dir) as f:
    lines = f.readlines()
    videomerged_tstamp_start = int(lines[0])
    print("video merged timestamp start at: ", videomerged_tstamp_start)
# read timestamp video
with open(video_tstamp_dir) as f:
    lines = f.readlines()
    index1 = lines[1].index(': ')
    index2 = lines[1].index(' mus')
    video_tstamp_start = int(lines[1][index1+2:index2])
    print("video timestamp start at: ", video_tstamp_start)

# the for cycle read the info from xml file, store them the in a pandas dataframe frame, and concatenate the dataframe
# to save all the info in the same dataframe df

for i, child in enumerate(root):

    for subchild in child:

        for subsubchild in subchild:

            a = 0 # do nothing

    frame = {}

    frame['SLOT'] = slot
    frame['PERSON'] = person
    frame['FRAME'] = child.find('timestamp').text
    #print(child.find('timestamp').text)
    frame['MILLISECONDS'] = child.find('milliseconds').text
    #print(child.find('milliseconds').text)
    frame['UTTERANCE'] = subchild.find('utterance').text
    frame['ADDRESSEE'] = subchild.find('addressee').text
    frame['TIMESTAMP_(mus)'] = video_tstamp_start + int(float(frame['MILLISECONDS'])) * 1000  # timestamp in microseconds
    frame['AUDIO_TIME_(mus)'] = frame['TIMESTAMP_(mus)'] + audio_tstamp_start
    frame['VIDEO_MERGED_(mus)'] = frame['TIMESTAMP_(mus)'] + videomerged_tstamp_start
    df_frame = pd.DataFrame(frame, index=[i])
    print(i)
    df = pd.concat([df, df_frame])

print(df)
df.to_csv(csv_dir, sep='\t')

