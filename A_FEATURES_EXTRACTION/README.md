# A Deep Learning Model for Addressee Estimation in Social Robots

Folder "A_FEATURES_EXTRACTION" contains codes to process data of Vernissage Corpus and create the dataset needed for the training of the Addressee Estimation Model.

The entire has to be followed for each interaction contained in the Vernissage Corpus separately.
Each file is a step of the data pipeline.
! Althought auditory features have been extracted, they have not been exploited to train the Addressee Estimation model, which (at the moment) only exploit visual information.

# 00_temporal_alignment.py

Brief description:
This code rewrites the information contained in the Annotation files into .csv files and align the timestamps of the annotation files with the ones of the three raw data files.

It takes in input: 
- Annotation files for each person of the Interaction (person1 and person2): "annotations_person{}.xml" 
- .txt files containing the timestamps of each raw data file (video and audio): "timestamp-naoaudio-begin.txt", 'timestamp-naoav-merged-begin.txt', 'naovideo-stats.txt'

It gives as output:
- .csv file with the annotation of the addressee, the time in sec., and the related timestamp for each raw data file: 'dataset_person{}.csv'

Python version & Requirements: Python 3.10, requirements_python10.txt

# 01_creation_dataset_pre_checking.py

Brief description:
This code creates clips of the original Vernissage Corpus, by trimming the raw data files with the time information of the csv file created by 00_temporal_alignment.py. The code must be run for each person of the interaction. 

It takes in input: 
- 'dataset_person{}.csv' created by 00_temporal_alignment.py
- original video merged file: 'naoav-merged.ogv' 
- original video file: 'naovideo.avi'
- original audio file: 'naoaudio.wav'

It gives as output:
- clips trimmed from the original raw data file. The most important are "{}videomerged{}.mp4", which contains both audio and video info merged.

Python version & Requirements: Python 3.10, requirements_python10.txt

# 02_creation_dataset_post_checking.py

Brief description:
This code creates extracts frames from video and audio clips, every 80 ms, and creates csv files for labels of the whole interaction (two persons), one for audio info, one for video info. .csv files contains info of 
- file frame name
- speaker
- addressee
- time of the raw data timeline each frame was saved (msec)
- time duration each frame is representative to (msec)
- n_interval each frame belongs to (i.e., number of clips each frames was extracted from) --> not chronological order

It takes in input: 
- clips trimmed from the merged video raw data: "{}videomerged{}.mp4". 
- clips trimmed from the audio raw data: "{}audio_trimmed{}.mp4". 
! clips taken in input are not all clips provided by 01_creation_dataset_pre_checking.py. Some of those clips have been removed after a manual check of files {}videomerged{}.mp4, to ensure that the clips contained correct information.

It gives as output:
- frames extracted from video '{}video_frame{}.jpg'
- mini-audio-clips trimmed from audio file '{}_audio_trimmed{}.wav'
- .csv file for video labels
- .csv file for audio labels

Python version & Requirements: Python 3.10, requirements_python10.txt

# 03_dataset_poses.py

Brief description:
This codes computes and extract body pose joints of people in video frames saved by "02_creation_dataset_post_checking.py". Body joints are extracted and saved both for the speaker and the other person present in the image. OpenPose algorithm is emplyed for the pose computation https://github.com/CMU-Perceptual-Computing-Lab/openpose (Coco version) 

It takes in input: 
- .jpg frames saved by '02_creation_dataset_post_checking.py'
- .csv video label file created by  '02_creation_dataset_post_checking.py

It gives as output:
- .npy files with info of the body pose joints for the speaker and the other
- .csv files 

Python version & Requirements: Python 3.6, requirements_python6.txt

# 03_dataset_poses.py

Brief description:
This codes computes and extract body pose joints of people in video frames saved by "02_creation_dataset_post_checking.py". Body joints are extracted and saved both for the speaker and the other person present in the image. OpenPose algorithm is emplyed for the pose computation https://github.com/CMU-Perceptual-Computing-Lab/openpose (Coco version) 

It takes in input: 
- .jpg frames saved by '02_creation_dataset_post_checking.py'
- .csv video label file created by  '02_creation_dataset_post_checking.py

It gives as output:
- .npy files with info of the body pose joints for the speaker and the other
- .csv files 

Python version & Requirements: Python 3.10, requirements_python10.txt

# 04_get_face_from_pose.py

Brief description:
This codes computes the face coordinates from the body joints and crop the image of the face from the video frame. The face is cropped both for the speaker and the other person in the image

It takes in input: 
- .jpg frames saved by '02_creation_dataset_post_checking.py'
- .npy files with info of the body pose joints for the speaker and the other from 03_dataset_poses.py

It gives as output:
- .jpg image of speaker's and other's face 

Python version & Requirements: Python 3.10, requirements_python10.txt

# 05_creation_label.py

Brief description:
This codes join the video and the audio information contained in the video and audio label file, creating a unique label file for each interaction with all the name of the files for each frame of interest.  It also checks that all the files are existent.

It takes in input: 
- labels_video_slot{}.csv (slot is the interaction)
- labels_audio_slot{}.csv 
- paths where frames, mini-audio-clips, body-joints, and face images are saved

It gives as output:
- 'all_labels.csv' file containing all the info for that interaction slot.

Python version & Requirements: Python 3.10, requirements_python10.txt

# 06_count_intervals.py

Brief description:
This codes divides frames in sequences of 10. Those sequences will be the ones used to train the hybrid model (CNN + LSTM). It also creates a new .csv label file with info about frames aggregated in sequences. It also re-check if there are missing data.

It takes in input: 

- 'all_labels.csv'

It gives as output:

- 'lstm_label_no_doubles.csv'

Python version & Requirements: Python 3.10, requirements_python10.txt

# 07_merging_dataset.py

Brief description:
This codes creates the final dataset with the files structured correctly to allow the training of the model.

It takes in input: 

- the path of the files where they have been saved so far
- the .jpg, .npy, .csv files contained in that folder and needed for the training

It gives as output:

- the files restructured in the new order. 

Python version & Requirements: Python 3.10, requirements_python10.txt

# 08_data_augmentation_flip.py

Brief description:
This codes augment data for LEFT and RIGHT classes to balance the number of occurrence of ROBOT class. Body Pose Joints and Face Images are flipped and consequently their label is changed. For instance, if a sequence of frames belongs to class LEFT, it is flipped and the outcome will belong to class RIGHT.

It takes in input: 

- 'lstm_label_no_doubles.csv'
- .npy files with info of the body pose joints for the speaker and the other from 03_dataset_poses.py
- .jpg image of speaker's and other's face from 04_get_face_from_pose.py

It gives as output:

- 'lstm_label_augmented_no_doubles.csv'

Python version & Requirements: Python 3.10, requirements_python10.txt

# 09_dataset_size.py

Brief description:
This codes plots the dimension of the dataset with respect to the three classes the model of this project has been trained with: 'NAO' ('ROBOT'), 'LEFT', 'RIGHT'

It takes in input: 

- 'lstm_label_{}_.csv'

Python version & Requirements: Python 3.10, requirements_python10.txt


## Support
If you need any support, please write an email to carlo.mazzola@iit.it https://orcid.org/0000-0002-9282-9873

## Authors and acknowledgment

Author of the code:
Carlo Mazzola https://orcid.org/0000-0002-9282-9873

Acknoweledgement:
Thanks to Marta Romeo (second author of the paper) for her support in designing the architecture and in several steps of the development 

## License
Creative Commons Attribution 4.0 International (CC-BY-4.0)
https://joinup.ec.europa.eu/licence/creative-commons-attribution-40-international-cc-40

For attribution, cite the paper 
C. Mazzola, M. Romeo, F. Rea, A. Sciutti and A. Cangelosi, "To Whom are You Talking? A Deep Learning Model to Endow Social Robots with Addressee Estimation Skills," 2023 International Joint Conference on Neural Networks (IJCNN), Gold Coast, Australia, 2023, pp. 1-10, 
https://doi.org/10.1109/IJCNN54540.2023.10191452



