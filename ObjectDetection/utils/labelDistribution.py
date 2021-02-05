import glob
import os

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

_path_data = 'data'

_typeSet = ['train', 'test', 'valid']

_cls = ['black-bishop', 'black-king', 'black-knight', 'black-pawn', 'black-queen', 'black-rook',
        'white-bishop', 'white-king', 'white-knight', 'white-pawn', 'white-queen', 'white-rook']

# Matplotlib utils
def autolabel(ax, rects, offset, percent=False):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.,
                h + offset, (f'{h}', f'{round(100*h)}%')[percent],
                fontsize='x-small',
                ha='center', va='bottom')

def save_bars_plot(data, legends, ylabel, path, title, percent=False):
    ind = np.arange(len(_cls))
    width = 0.8 / len(legends)

    fig, ax = plt.subplots()
    ax.set_ylabel(ylabel)
    ax.set_xticks(ind + (0.8 - width) / 2)
    ax.set_xticklabels(_cls)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize='x-small')

    if percent:
        counts = [[x / data[label]['tot_boxes'] for x in data[label]['count']] for label in legends]
    else:
        counts = [data[label]['count'] for label in legends]

    offset_text = 0.01 * max([max(counts[i]) for i in range(len(legends))])

    rects = []
    for i in range(len(legends)):
        rect = ax.bar(
                ind + width * i,
                counts[i],
                width=width)
        autolabel(ax, rect, offset_text, percent=percent)
        rects.append(rect)

    ax.legend(rects, legends)
    plt.title(path[path.find(_path_data)+len(_path_data)+1:])

    fig.savefig(f'{path}/{title}.png')
    plt.close(fig)
    print(f'  {path}/{title}.png saved...')

def create_data_counter(path):
    data_counter = {}
    data_counter['path_to_data'] = path
    data_counter['nb_files'] = 0
    data_counter['tot_boxes'] = 0
    data_counter['count'] = [0 for i in range(len(_cls))]
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

def merge_data_folder(data_folder, path):
    merged_counter = create_data_counter(path)

    for subdir, counter in data_folder.items():
        merged_counter['nb_files'] += counter['nb_files']
        merged_counter['tot_boxes'] += counter['tot_boxes']
        for i in range(len(_cls)):
            merged_counter['count'][i] += counter['count'][i]

    return merged_counter

def analyse_folder(path):
    data_folder = {}

    plt_labels=[]

    for subdirectory in _typeSet:
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

        plt_labels.append(subdirectory)

    if len(plt_labels) > 0:
        # Percentage figure
        save_bars_plot(data=data_folder, legends=plt_labels,
                       ylabel='% of appearance within each set',
                       path=path, title='labels_distribution_percent',
                       percent=True)

        # Number figure
        save_bars_plot(data=data_folder, legends=plt_labels,
                       ylabel='# of appearances within each set',
                       path=path, title='labels_distribution',
                       percent=False)

    return data_folder

if __name__ == '__main__':
    if not os.path.exists(_path_data):
        print(f'The path to the data "{_path_data}" doesn\'t exists...')
        exit()

    data_all = {}
    plt_directories = []

    for directory in os.listdir(_path_data):
        path_directory = os.path.join(_path_data, directory)

        # Ignore if this is not a folder
        if not os.path.isdir(path_directory):
            continue

        print(f'Directory {path_directory}...')

        data_folder = analyse_folder(path_directory)
        if len(data_folder) > 0:
            data_all[directory] = merge_data_folder(data_folder, path_directory)
            plt_directories.append(directory)

        print(f'Directory {path_directory}... Done\n')

    print('Saving global data...')

    # Percentage figure
    save_bars_plot(data=data_all, legends=plt_directories,
                   ylabel='% of appearance within each directories',
                   path=_path_data, title='labels_distribution_percent',
                   percent=True)

    # Number figure
    save_bars_plot(data=data_all, legends=plt_directories,
                   ylabel='# of appearances within each directories',
                   path=_path_data, title='labels_distribution',
                   percent=False)

    print('Done')
