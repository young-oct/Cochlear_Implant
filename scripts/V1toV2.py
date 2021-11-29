# -*- coding: utf-8 -*-
# @Time    : 2021-11-29 8:56 a.m.
# @Author  : young wang
# @FileName: V1toV2.py
# @Software: PyCharm
"""convert V1 header 3D bin file to V2 header oct file"""

import numpy as np
from OssiviewBufferReader import OssiviewBufferReader
import os
from os.path import join
import string
import glob


def importfV1(src_path):
    bin_obj = OssiviewBufferReader(src_path).data['3D Buffer']
    data = 20 * np.log10(abs(bin_obj))
    return data.reshape((1, 512, 512, 330))


def export2V2(data, dis_path):
    bin_obj = OssiviewBufferReader(dis_path)
    bin_obj.updateDataV2(data)
    bin_obj.exportV2(dis_path)
    return None


if __name__ == '__main__':


    bin_files = []
    directory = '/Users/youngwang/Desktop/CI_Ossiview'

    for filepath in glob.iglob(r'/Users/youngwang/Desktop/CI_Ossiview/*.bin'):
        bin_files.append(filepath)

    bin_files.sort()
    #
    prefix_path = '/Users/youngwang/Desktop/CI_OCT'

    # src_path = '/Users/youngwang/Desktop/CI_Ossiview/2021-Nov-08_08.02.19_AM_Bin_Capture.oct'

    dis_path = list(string.ascii_lowercase)[0:len(bin_files)]
    export_paths= []
    for i in range(0,len(dis_path)):

        path = join(prefix_path,dis_path[i])
        export_paths.append(path)

        if not os.path.exists(path):
            os.mkdir(path)
        else:
            pass
    #
        src_path = bin_files[i]
        data = importfV1(src_path)
        file_name = dis_path[i] + '.oct'
        # print(path, file_name)
        output_path = join(path,file_name)
        # print(output_path)
        export2V2(data, output_path)
