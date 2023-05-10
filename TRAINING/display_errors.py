import os
import numpy as np
import pandas as pd
import cv2 as cv
from test import test
import matplotlib.pyplot as plt


def show_sequence(images, label, prediction, sequence):

    # create figure
    fig = plt.figure(figsize=(6, 1))
    description = 'REAL class was {} ---> PREDICTED class is {}'.format(d['class_names'][label], d['class_names'][prediction])
    plot_dir = os.path.join(plot_path, 'sequence_{}_{}_{}_{}.jpg'.format(
        slot, sequence, d['class_names'][label], d['class_names'][prediction]))
    # setting values to rows and column variables
    rows = 1
    columns = 10
    plt.text(0, 0, description)
    plt.axis('off')
    for n, image in enumerate(images):
        image_dir = os.path.join(csv_path, slot, 'face_s', image)
        frame = cv.imread(image_dir)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # Adds a subplot
        fig.add_subplot(rows, columns, n+1)

        # showing image
        plt.imshow(frame)
        plt.axis('off')
        plt.title(str(n))

    #plt.show()
    plt.savefig(plot_dir)
    plt.close()


model_name = 'lstm_vision'
version = 'v8_30'
model_file = '{}_info.npy'.format(model_name)
save_dir = '/home/icub/Documents/Carlo/models/FINALS/cnn_lstm_mm'
save_dir = os.path.join(save_dir, version)
n_slot = 9
models_list = [model for model in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, model))]
models_list.sort()

csv_path = '/home/icub/Documents/Carlo/dataset_slots'
slots_list = [name for name in os.listdir(csv_path) if os.path.isdir(os.path.join(csv_path, name))]
slots_list.sort()
slot = slots_list[n_slot]
test_filename = 'test_slot_{}_out.csv'.format(slot)

save_dir = os.path.join(save_dir, models_list[n_slot])

dict_dir = os.path.join(save_dir, model_file)

d, error_df = test(dict_dir, test_filename)
class_names = d['class_names']
plot_path = os.path.join('/home/icub/Documents/Carlo/models/PLOTS/sequences_failed', model_name, version)
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

test_df = pd.read_csv(os.path.join(csv_path, test_filename), sep='\t', index_col=False)

for i, sequence in enumerate(error_df['sequence'].values):
    images = test_df[test_df['SEQUENCE'] == int(sequence)]['FACE_SPEAKER'].values
    show_sequence(images, error_df['label'].values[i], error_df['prediction'].values[i], sequence)

