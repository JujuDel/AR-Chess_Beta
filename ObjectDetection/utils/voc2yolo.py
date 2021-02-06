import os
import glob
import argparse

import xml.etree.ElementTree as ET

from tqdm import tqdm

# Read and parse the user input arguments
def read_args():
    parser = argparse.ArgumentParser(description='PascalVOC to YOLO format converter.')
    parser.add_argument('--label', type=str, required=True,
                        help='directory of label folder or label file path')
    parser.add_argument('--output', type=str, required=True,
                        help='output directory')
    parser.add_argument('--cls', type=str, required=True,
                        help='txt file containing the list of the labels')

    return parser, parser.parse_args()

# Check the validity of the arguments
def check_args(args):
    ret = True

    if not os.path.exists(args.label):
        print(f'/!\\ Error: the label path "{args.label}" is neither a file nor a folder...')
        ret = False
    elif os.path.isfile(args.label) and args.label[-4:] != '.xml':
        print(f'/!\\ Error: the label file "{args.label}" should be a xml file...')
        ret = False

    if not os.path.isfile(args.cls):
        print(f'/!\\ Error: the given cls "{args.cls}" is not a file...')
        ret = False
    elif args.cls[-4:] != '.txt':
        print(f'/!\\ Error: the given cls "{args.cls}" is not a txt file...')
        ret = False

    return ret

# Read and convert one xml file in PascalVOC format to one txt file in YOLO format
def convert_annotation(path_in, folder_out, labels):
    try:
        in_file = open(path_in, encoding='utf-8')
    except (OSError, IOError) as e:
        raise e

    tree = ET.parse(in_file)
    root = tree.getroot()

    try:
        name = root.find('filename').text
        for ext in ['.png', '.jpg']:
            if ext in name:
                name = name[:name.find(ext)]

        out_file = open(os.path.join(folder_out, f'{name}.txt'), 'w')

        width = float(root.find('./size/width').text)
        height = float(root.find('./size/height').text)

        for i, obj in enumerate(root.iter('object')):
            cls = obj.find('name').text

            if cls not in labels:
                continue
            cls_id = labels.index(cls)

            xmlbox = obj.find('bndbox')

            xmin = float(xmlbox.find('xmin').text) / width
            xmax = float(xmlbox.find('xmax').text) / width
            ymin = float(xmlbox.find('ymin').text) / height
            ymax = float(xmlbox.find('ymax').text) / height

            if i != 0:
                out_file.write('\n')
            out_file.write(f'{cls_id} {xmin} {ymin} {xmax} {ymax}')

        out_file.close()
    except:
        print('FUCK')
        out_file.close()
        raise


def main(args):
    # Prepare a list of all the files to convert
    if os.path.isfile(args.label):
        all_files = [args.label]
    else:
        all_files = glob.glob(os.path.join(args.label, '*.xml'))

    # Prepare the output directory
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Prepare the label list
    with open(args.cls, 'r') as file:
        args.cls = file.read().split('\n')

    # Loop through the files to convert
    notFound = []
    for path_in in tqdm(all_files):
        name = path_in.replace('\\', '/')
        while '/' in name:
            name = name[name.find('/')+1:]
        name = name[:name.find('.xml')]

        try:
            convert_annotation(path_in=path_in, folder_out=args.output, labels=args.cls)
        except:
            notFound.append(path_in)

    if len(notFound) > 0:
        print(f'/!\\ Warning: error while loading {len(notFound)}/{len(all_files)} file{("","s")[len(all_files)>1]} /!\\')
        for path in notFound:
            print(f'\t{path}')


if __name__ == '__main__':
    parser, args = read_args()

    if not check_args(args=args):
        print()
        parser.print_help()
        exit()

    main(args)
    print('DONE')