# -*- coding: utf-8 -*-
# @Time    : 2021-11-30 8:47 a.m.
# @Author  : young wang
# @FileName: otoscopeImage.py
# @Software: PyCharm


from matplotlib import pyplot as plt
import numpy as np
import glob
from os.path import join
import cv2


def circleROI(img, radius):
    _mask = np.zeros(img.shape, dtype=np.uint8)

    x, y = img.shape[1], img.shape[0]

    cv2.circle(_mask, (int(x / 2), int(y / 2)),
               radius, (255, 255, 255), -1)

    _ROI = cv2.bitwise_and(img, _mask)
    _mask = cv2.cvtColor(_mask, cv2.COLOR_BGR2GRAY)
    x, y, w, h = cv2.boundingRect(_mask)
    result = _ROI[y:y + h, x:x + w]
    _mask = _mask[y:y + h, x:x + w]
    result[_mask == 0] = (255, 255, 255)

    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':


    input_folder = '../Figures/cadaver_otoscope'
    path = join(input_folder, '*.png')
    filenames = [filename for filename in glob.glob(path)]
    filenames.sort(reverse=False)

    output_path = '../Figures/cadaver_otoscope/processed/'

    title = ['full insertion', 'partial insertion 2', 'partial insertion 3', 'full migration']

    for _ in range(len(filenames)):
        oto_image = cv2.imread(filenames[_])
        oto_image = circleROI(oto_image, 360)

        fig, ax = plt.subplots(1,1, figsize=(10,10))

        ax.imshow(oto_image)
        ax.axis('off')
        plt.tight_layout()
        file_name = output_path + title[_] + '.jpeg'
        plt.savefig(file_name,
                    dpi=800,
                    transparent=True, bbox_inches='tight', pad_inches=0, format='jpeg')



        plt.show()
    # sorted(filenames,reverse=True)
    # otc_image = plt.imread(OCT_files[i])
