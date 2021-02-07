import glob
import os

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

_path_data = 'data'

_typeSet = ['.', 'train', 'test', 'valid']

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

# Matplotlib scatter plot
def save_WxH_plot(data, subdirs):
    # For all the subdirs
    for subdir in subdirs:
        # Create the figure
        fig, ax = plt.subplots()

        # Scatter the data for all the pieces
        for i in range(len(_cls)):
            ax.scatter(data[subdir]['W'][i], data[subdir]['H'][i], label=_cls[i], s=8)

        # Find the limits of the graph
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        # Plot the line y = x
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        # Add the x and y labels
        plt.ylabel('H')
        plt.xlabel('W')
        # Show the legends
        plt.legend()

        # Adapt the path
        path = data[subdir]['path_to_data']
        if '/labels' in path or '\\labels' in path:
            path = path[:path.find('labels')-1]
        if '/.' in path:path = path[:path.find('/.')]
        if '\\.' in path:path = path[:path.find('\\.')]

        # Add the title
        title = path.replace('\\', '/')
        for tmp in _typeSet:
            if tmp in path:
                title = tmp
                break
        while '/' in title:
            title = title[title.find('/')+1:]
        plt.title(title)

        final_path = os.path.join(path,'sizes_'+title+'.png')
        # Save the figure
        fig.savefig(final_path)
        plt.close(fig)
        print(f'  {final_path} saved...')

# Matplotlib bars plot
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

# All the counter info in a txt file
def save_bars_txt(data, path, title):
    # Extact the name of the folder
    name = path[path.find(_path_data)+len(_path_data)+1:].replace('\\', '/')
    if '/' in name: name = name[:name.find('/')]

    maxLen = len(max(_cls, key=lambda x:len(x))) + 2

    # Merge the inner data together
    merged_data = merge_data_folder(data, name)

    with open(os.path.join(path,f'{title}.txt'), 'w') as file:
        if name != '': file.write(f'{name}:\n')
        file.write(f'\tnb_files........: {merged_data["nb_files"]}\n')
        file.write(f'\ttot_boxes.......: {merged_data["tot_boxes"]}\n')
        for i in range(len(_cls)):
            file.write(f'\t\t{_cls[i]}' +
                       '.'*(maxLen-len(_cls[i])) +
                       f': {merged_data["count"][i]}' +
                       '\t{:.2f}%\n'.format(100 * merged_data["count"][i] / merged_data["tot_boxes"]))

        for subdir, data_analyser in data.items():
            file.write(f'{subdir}:\n')
            file.write(f'\tpath_to_labels..: {data_analyser["path_to_data"]}\n')
            file.write(f'\tnb_files........: {data_analyser["nb_files"]}\n')
            file.write(f'\ttot_boxes.......: {data_analyser["tot_boxes"]}\n')
            for i in range(len(_cls)):
                file.write(f'\t\t{_cls[i]}' +
                           '.'*(maxLen-len(_cls[i])) +
                           f': {data_analyser["count"][i]}' +
                           '\t{:.2f}%\n'.format(100 * data_analyser["count"][i] / data_analyser["tot_boxes"]))

# Default dictionnary which holds data analysers
def create_data_analyser(path):
    data_analyser = {}
    data_analyser['path_to_data'] = path
    data_analyser['nb_files'] = 0
    data_analyser['tot_boxes'] = 0
    data_analyser['count'] = [0 for i in range(len(_cls))]
    data_analyser['W'] = [np.array([]) for i in range(len(_cls))]
    data_analyser['H'] = [np.array([]) for i in range(len(_cls))]
    return data_analyser

# Analyse the label within a folder of txt files in yolov5 pytorch format
def analyse_labels(path_labels):
    data_analyser = create_data_analyser(path_labels)

    for path_txt in tqdm(glob.glob(f'{path_labels}/*.txt')):
        data_analyser['nb_files'] += 1
        with open(path_txt) as file:
            lines = file.read().split('\n')

        for line in lines:
            if len(line) < 1: continue

            label, xmin, ymin, width, height = map(float, line.split())
            label = int(label)

            data_analyser['count'][label] += 1
            data_analyser['tot_boxes'] += 1
            data_analyser['W'][label] = np.append(data_analyser['W'][label], width)
            data_analyser['H'][label] = np.append(data_analyser['H'][label], height)

    return data_analyser

# Merge the analyser of the data_folder holder
def merge_data_folder(data_folder, path):
    merged_analyser = create_data_analyser(path)

    for subdir, analyser in data_folder.items():
        merged_analyser['nb_files'] += analyser['nb_files']
        merged_analyser['tot_boxes'] += analyser['tot_boxes']
        for i in range(len(_cls)):
            merged_analyser['count'][i] += analyser['count'][i]
            merged_analyser['W'][i] = np.append(merged_analyser['W'][i], analyser['W'][i])
            merged_analyser['H'][i] = np.append(merged_analyser['H'][i], analyser['H'][i])

    return merged_analyser

# Analyse the given folder by checking its subdirectories
def analyse_folder(path):
    # Holder of the data info in the folder
    data_folder = {}

    # List of the _typeSet subdirectories analysed
    plt_labels = []

    # Go through all the _typeSet subdirectories
    for subdirectory in _typeSet:
        path_subdirectory = os.path.join(path_directory, subdirectory)

        # Ignore if the subdirectory doesn't exists
        if not os.path.exists(path_subdirectory):
            continue

        path_subdirectory_labels = os.path.join(path_subdirectory, 'labels')
        if not os.path.exists(path_subdirectory_labels):
            print(f'  - Subdirectory "{subdirectory}/" doesn\'t contain a folder "labels"! Ignoring it...')
            continue

        print(f'  - "{subdirectory}/"...')

        # Count the labels of this subdirectory
        data_folder[subdirectory] = analyse_labels(path_subdirectory_labels)

        # This subdirectory has been analysed
        plt_labels.append(subdirectory)

    if len(plt_labels) > 0:
        # Sizes figure
        save_WxH_plot(data=data_folder, subdirs=plt_labels)

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

        # Dump the data
        save_bars_txt(data=data_folder, path=path,
                      title='labels_distribution')

    return data_folder

if __name__ == '__main__':
    if not os.path.exists(_path_data):
        print(f'The path to the data "{_path_data}" doesn\'t exists...')
        exit()

    # Holder of all the data info
    data_all = {}

    # List of the subdirectories with labels inside
    plt_directories = []

    for directory in os.listdir(_path_data):
        path_directory = os.path.join(_path_data, directory)

        # Ignore if this is not a folder
        if not os.path.isdir(path_directory):
            continue

        print(f'Directory {path_directory}...')

        # Analyse the folder
        data_folder = analyse_folder(path_directory)
        if len(data_folder) > 0:
            # This folder contains info, merge the analyser
            data_all[directory] = merge_data_folder(data_folder, path_directory)
            plt_directories.append(directory)

        print(f'Directory {path_directory}... Done\n')

    print('Saving global data...')

    # Sizes figure
    save_WxH_plot(data=data_all, subdirs=plt_directories)

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

    # Dump the data
    save_bars_txt(data=data_all, path=_path_data,
                  title='labels_distribution')

    print('Done')
