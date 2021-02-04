import glob
import os

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

path_data = 'data'

typeSet = ['train', 'test', 'valid']

cls = ['black-bishop', 'black-king', 'black-knight', 'black-pawn', 'black-queen', 'black-rook',
       'white-bishop', 'white-king', 'white-knight', 'white-pawn', 'white-queen', 'white-rook']

# Matplotlib utils
def autolabel(ax, rects, offset):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.,
                h + offset, f'{round(100*h)}%',
                fontsize='x-small',
                ha='center', va='bottom')

def create_data_counter(path):
    data_counter = {}
    data_counter['path_to_data'] = path
    data_counter['nb_files'] = 0
    data_counter['tot_boxes'] = 0
    data_counter['count'] = [0 for i in range(len(cls))]
    return data_counter

def count_labels(path_labels):
    data_counter = create_data_counter(path_labels)
    
    for path_txt in tqdm(glob.glob(f'{path_labels}/*.txt')):
        data_counter['nb_files'] += 1
        with open(path_txt) as file:
            lines = file.read().split('\n')
        
        for line in lines:
            if len(line) < 1: continue
            label = int(line.split()[0])
            data_counter['count'][label] += 1
            data_counter['tot_boxes'] += 1

    return data_counter

def analyse_folder(path):
    data_folder = {}

    fig, ax = plt.subplots()
    ind = np.arange(len(cls))

    rects=[]
    plt_labels=[]

    offset_text = 0
        
    for i, subdirectory in enumerate(typeSet):
        path_subdirectory = os.path.join(path_directory, subdirectory)

        # Ignore if the subdirectory doesn't exists
        if not os.path.exists(path_subdirectory):
            continue

        path_subdirectory_labels = os.path.join(path_subdirectory, 'labels')
        if not os.path.exists(path_subdirectory_labels):
            print(f'  - Subdirectory "{subdirectory}" doesn\'t contain a folder "labels"! Ignoring it...')
            continue

        print(f'  {subdirectory}...')

        data_folder[subdirectory] = count_labels(path_subdirectory_labels)
        
        data_folder[subdirectory]['count'] = [x / data_folder[subdirectory]['tot_boxes'] for x in data_folder[subdirectory]['count']]

        offset_text = max(offset_text, max(data_folder[subdirectory]['count']))
        rects.append(ax.bar(ind + 0.2 * i, data_folder[subdirectory]['count'], width=0.2))
        plt_labels.append(subdirectory)
    
    if len(rects) > 0:
        ax.set_ylabel('% of appearances within each set')
        ax.set_xticks(ind + 0.2)
        ax.set_xticklabels(cls)
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right',fontsize='x-small')

        offset_text = 0.01 * offset_text
        
        for rect in rects:
            autolabel(ax, rect, offset_text)
    
        ax.legend(rects, plt_labels)
        plt.title(path[path.find(path_data)+len(path_data)+1:])
        
        fig.savefig(f'{path}/labels_distribution.png')
        plt.close(fig)
        print(f'  {path}/labels_distribution.png saved...')

    return data_folder

if __name__ == '__main__':
    if not os.path.exists(path_data):
        print(f'The path to the data "{path_data}" doesn\'t exists...')
        exit()

    for directory in os.listdir(path_data):
        path_directory = os.path.join(path_data, directory)

        # Ignore if this is not a folder
        if not os.path.isdir(path_directory):
            continue

        print(f'Directory {path_directory}...')

        data_directory = analyse_folder(path_directory)

        print(f'Directory {path_directory}... Done\n')
