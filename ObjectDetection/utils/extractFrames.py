import cv2
import os

from tqdm import tqdm

import numpy as np

# Expected input of the form hh:mm:ss
def str_to_ms(timestamp):
    h, m, s = map(int, timestamp.split(':'))
    
    return 1000 * (3600 * h + 60 * m + s)

def resize_to_square(img, s=416):
    h,w,d = img.shape
    
    M = max(w, h)
    dark = np.zeros((M,M,3),np.uint8)
    
    cx = round((M - w) / 2)
    cy = round((M - h) / 2)
    
    dark[cy:h+cy,cx:w+cx] = img
    
    if w == h:
        return cv2.resize(img, (s,s))
    
    if w > h:
        diff = round((w - h) / 2)
        img_cropped = img[:, diff:w-diff]
    else:
        diff = round((h - w) / 2)
        img_cropped = img[diff:h-diff, :]
    
    return cv2.resize(dark, (s,s))

if __name__ == '__main__':
    folder = '1_CHECKMATE_2018_Finals'

    video = cv2.VideoCapture(f'data/{folder}/video.mp4')
    fps = int(video.get(cv2.CAP_PROP_FPS))
    
    with open(f'data/{folder}/timestamps.txt') as f:
        timestamps = f.read().split('\n')
    
    if not os.path.exists(f'data/{folder}/images'):
        os.makedirs(f'data/{folder}/images')
    if not os.path.exists(f'data/{folder}/images_416x416'):
        os.makedirs(f'data/{folder}/images_416x416')
    
    for timestamp in tqdm(timestamps):
        ms = str_to_ms(timestamp)
        video.set(cv2.CAP_PROP_POS_MSEC, ms)
        
        ret, frame = video.read()
        if ret == False:
            break
            
        cv2.imwrite(f'data/{folder}/images/{folder}_{ms}.png', frame)
        cv2.imwrite(f'data/{folder}/images_416x416/{folder}_{ms}.png', resize_to_square(frame))
    
    video.release()
    cv2.destroyAllWindows()