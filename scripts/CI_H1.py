# -*- coding: utf-8 -*-
# @Time    : 2021-11-29 8:14 a.m.
# @Author  : young wang
# @FileName: CI_H1.py
# @Software: PyCharm

'''this script removes the top and bottom artfacts from the volumne bin file using H1
header'''

# -*- coding: utf-8 -*-
# @Time    : 2021-11-26 7:35 p.m.
# @Author  : young wang
# @FileName: V1todicom.py
# @Software: PyCharm

import pydicom
import numpy as np
from OssiviewBufferReader import OssiviewBufferReader
from os.path import join, isfile
from skimage.filters import median
from skimage.morphology import cube
import string
import os


def loadExport(input_file):
    obr = OssiviewBufferReader(input_file)
    data = obr.data['3D Buffer'].squeeze()

    data = 20 * np.log10(abs(data))

    data = imag2uint(data)

    data = clean(data)

    return median(data, cube(3))





def clean(data):
    top = 30
    data[:, :, 0:top] = 0
    # data[:,:,256] = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.sqrt((i - 256) ** 2 + (j - 256) ** 2) >= 230:
                data[i, j, :] = 0

    return data



if __name__ == '__main__':

    oct_files = []
    directory = '/Users/youngwang/Desktop/CI_bin'
    import glob

    for filepath in glob.iglob(r'/Users/youngwang/Desktop/CI_bin/*.bin'):
        oct_files.append(filepath)

    oct_files.sort()

    prefix_path = '/Users/youngwang/Desktop/DICOM Export'
    dis_path = list(string.ascii_lowercase)[0:len(oct_files)]
    export_paths = []
    for i in range(len(dis_path)):

        path = join(prefix_path, dis_path[i])
        export_paths.append(path)

        if not os.path.exists(path):
            os.mkdir(path)
        else:
            pass

    dicom_prefix = 'CI-cadaver'
    seriesdescription = ['Full Insertion',
                         'Partial Insertion I',
                         'Partial Insertion II',
                         'Full Withdrawal I',
                         'Full Withdrawal II',
                         'Full Withdrawal III']

    for i in range(len(oct_files)):
        export_path = join(prefix_path, dis_path[i])

        if not os.path.exists(export_path):
            os.mkdir(export_path)
        else:
            pass

