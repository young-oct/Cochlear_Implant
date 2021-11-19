# -*- coding: utf-8 -*-
# @Time    : 2021-11-19 8:18 a.m.
# @Author  : young wang
# @FileName: frame_exporter.py
# @Software: PyCharm
import os.path

import cv2
from matplotlib import pyplot as plt


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

    title = ['full insertion', 'partial insertion 2', 'partial insertion 3', 'full migration']
    for _ in range(len(a)):
        plt.imshow(a[_], 'gray', aspect=a[_].shape[1] / a[_].shape[0])
        # plt.title(title[_])
        plt.axis('off')
        plt.tight_layout()
        file_name = output_path + title[_] + '.jpeg'
        plt.savefig(file_name,
                    dpi=800,
                    transparent=True, bbox_inches='tight', pad_inches=0, format='jpeg')
        plt.show()
