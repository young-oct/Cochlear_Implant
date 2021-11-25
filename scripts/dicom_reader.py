# -*- coding: utf-8 -*-
# @Time    : 2021-11-24 9:39 a.m.
# @Author  : young wang
# @FileName: dicom_reader.py
# @Software: PyCharm
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from skimage import data, img_as_float,img_as_uint,img_as_ubyte
from scipy import ndimage, misc

from skimage import exposure


if __name__ == '__main__':

    input_file = '../Cochlear_Implant/Figures/3487.dcm'
    output_path = '../Cochlear_Implant/Figures/patient_2D/'

    ds = pydicom.dcmread(input_file)

    img = ds.pixel_array
    img = np.rot90(img)
    img = np.flip(img,1)
    start = 20
    end = 800
    width = end - start

    img = img_as_float(img)
    img = exposure.equalize_adapthist(img, clip_limit=0.015)
    img = ndimage.median_filter(img, size=3)

    img =img_as_ubyte(img)

    plt.imshow(img[:,start:end],'gray', aspect=width/img.shape[0], vmin = 20, vmax=245)
    plt.axis('off')
    plt.tight_layout()
    # title = 'CI(patient)'
    # file_name = output_path + title + '.jpeg'
    # plt.savefig(file_name,
    #             dpi=800,
    #             transparent=True, bbox_inches='tight', pad_inches=0, format='jpeg')
    plt.show()
