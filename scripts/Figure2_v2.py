# -*- coding: utf-8 -*-
# @Time    : 2021-12-10 11:01 a.m.
# @Author  : young wang
# @FileName: Figure2_v2.py
# @Software: PyCharm


import matplotlib.pyplot as plt
import matplotlib
from os.path import join, abspath
import os
from skimage import img_as_float, img_as_ubyte
from skimage import data, exposure, img_as_float
from skimage import morphology
from skimage.morphology import square,disk
from skimage.filters import median
import glob
import numpy as np
np.seterr(invalid='ignore')

if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 20,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    input_path = '../Figures/CI pictures'

    # img_files = sorted(os.listdir(output_path))
    path = input_path+('/*.jpg')
    img_paths = []
    for filename in glob.glob(path):
        img_paths.append(filename)

    img_paths.sort()
    y1 , y_height = 250, 500
    x1 , x_width = 550, 500
    title = ['full insertion', 'two-marker', 'three-marker','full withdrawal']
    #
    export_path = '../Figures/cadaverCI_enhanced'
    extension = '.jpeg'

    join(export_path,title[0])
    for i in range(len(img_paths)):

        fig,ax = plt.subplots(1,1)
        img = plt.imread(img_paths[i])
        image = img[y1:int(y1+y_height), x1:int(x1+x_width),0]

        image = exposure.equalize_adapthist(img, clip_limit=0.009)
        image = img_as_ubyte(image)
        # image = median(image,square(3))
        # image = morphology.closing(image)
        image = exposure.adjust_gamma(image, 0.8)

        ax.imshow(image, aspect=image.shape[1]/image.shape[0],
                   vmin=0, vmax = 255)
        # ax.axis('off')
        plt.tight_layout()
        plt.show()

    #
    #     file_name = join(export_path,title[i]) + extension
    #     plt.savefig(file_name,
    #                 dpi=800,
    #                 transparent=True, bbox_inches='tight', pad_inches=0, format='jpeg')
    #     plt.show()
