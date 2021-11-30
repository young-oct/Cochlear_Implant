# -*- coding: utf-8 -*-
# @Time    : 2021-11-19 8:18 a.m.
# @Author  : young wang
# @FileName: frame_exporter.py
# @Software: PyCharm
import os.path

import cv2
from matplotlib import pyplot as plt
from skimage.morphology import square,disk
from skimage.filters import median
import numpy as np
from skimage import exposure
from skimage import img_as_float, img_as_ubyte
from skimage import data, exposure, img_as_float


def video2frame():
    try:
        cap = cv2.VideoCapture(input_file)
    except IOError:
        print("File not accessible")
    return cap


def get_frame(video_data, time_stamp):
    data = video_data.data
    info = video_data.info
    fps = info['fps']
    max_frame = info['total frame']

    def timeframe(hour=0, minute=0, second=0, fps=0):
        specific_frame = (hour * 3600 + minute * 60 + second) * int(fps)
        return int(specific_frame)

    index = []
    for i in range(len(time_stamp)):
        cur_time = time_stamp[i]
        idx = timeframe(cur_time[0], cur_time[1], cur_time[2], fps=fps)
        if idx <= max_frame:
            index.append(idx)
        else:
            raise Exception('out of bounds')

    frames = []
    for _ in range(len(index)):
        data.set(1, index[_])
        _, frame = data.read()
        frames.append(frame[0: 650, 225: 650, 0])
    return frames


class video:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = video2frame()
        self.info = self.getInfo()

    def getInfo(self):
        frame_width = int(self.data.get(3))
        frame_height = int(self.data.get(4))
        fps = self.data.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.data.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        info = {'width': frame_width, 'height': frame_height,
                'fps': fps, 'total frame': frame_count, 'duration': duration}
        return info


if __name__ == '__main__':
    input_file = '../Video/media1.mp4'
    output_path = '../Figures/cadaver_2D/'
    record = video(input_file)

    time_stamp = [[0, 0, 0.1], [0, 0, 2.5], [0, 0, 4.5], [0, 0, 8]]

    a = get_frame(record, time_stamp)

    try:
        os.mkdir(output_path)
    except OSError:
        pass

    from skimage import morphology

    title = ['full insertion', 'partial insertion 2', 'partial insertion 3', 'full migration']
    for _ in range(len(a)):
        temp = img_as_float(a[_])
        temp = median(temp,square(3))

        # img_rescale =  exposure.adjust_gamma(temp, 1.1)

        img_rescale = exposure.equalize_adapthist(temp, clip_limit=0.025)
        #
        img_rescale = img_as_ubyte(img_rescale)
        #
        img_rescale = median(img_rescale, square(3))

        # img_rescale = morphology.opening(img_rescale,disk(10))

        plt.imshow(img_rescale, 'gray', aspect=a[_].shape[1] / a[_].shape[0], vmin=50,vmax =250)
        # plt.title(title[_])
        plt.axis('off')
        plt.tight_layout()
        file_name = output_path + title[_] + '.jpeg'
        plt.savefig(file_name,
                    dpi=800,
                    transparent=True, bbox_inches='tight', pad_inches=0, format='jpeg')
        plt.show()

    # plt.hist(img_rescale.flatten())
    # plt.show()
