# -*- coding: utf-8 -*-
# @Time    : 2021-11-26 2:38 p.m.
# @Author  : young wang
# @FileName: header_conversion.py
# @Software: PyCharm

import numpy as np
from OssiviewBufferReader import OssiviewBufferReader
from os.path import join, isfile
from pydicom.uid import generate_uid
import matplotlib.pyplot as plt

def load_from_oct_file(oct_file):
    """
    read .oct file uising OssiviewBufferReader
    export an array in the shape of [512,512,330]
    the aarry contains pixel intensity data(20log) in float16 format
    """
    obr = OssiviewBufferReader(oct_file)
    data_fp16 = np.squeeze(obr.data)
    return data_fp16

def loadV1(input_file):
    obr = OssiviewBufferReader(input_file)
    data = obr.data['3D Buffer'].squeeze()
    data = 20*np.log10(abs(data))
    data = data.astype(np.float16)


    # data = np.swapaxes()
    V2_data = data[np.newaxis,:]
    return V2_data

def write_oct_file(template_file, data):

    obr = OssiviewBufferReader(template_file)
    obr.updateDataV2(data)
    obr.exportV2(template_file)
    return None


if __name__ == '__main__':

    input_file ='/Users/youngwang/Desktop/CI_P/2021-Jul-13  03.28.27 PM.bin'

    data = loadV1(input_file)

    template_file = '/Users/youngwang/Desktop/2021-Nov-08_08.02.19_AM_Bin_Capture.oct'
    write_oct_file(template_file, data)

    # a = load_from_oct_file(template_file)


