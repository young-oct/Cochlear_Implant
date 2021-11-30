# -*- coding: utf-8 -*-
# @Time    : 2021-11-24 11:06 a.m.
# @Author  : young wang
# @FileName: Figure2.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import matplotlib

if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 20,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    output_path = '../Cochlear_Implant/Figures/'

    img1 = plt.imread('../Cochlear_Implant/Figures/patient_2D/otoscope.png')
    img2 = plt.imread('../Cochlear_Implant/Figures/patient_2D/CI(patient).png')
    img3 = plt.imread('../Cochlear_Implant/Figures/patient_2D/CI_3D.png')

    images = [img1,img2,img3]

    fig,axs = plt.subplots(nrows = 1, ncols =3, figsize=(16,9))

    titles = ['(A) Otoscopic Image', '(B) 2D OCT', '(C) 3D OCT']

    for img,ax,title in zip(images, axs.flatten(),titles):
        ax.imshow(img, aspect=img.shape[1]/img.shape[0])
        ax.axis('off')
        ax.set_title(title, weight='bold')
    plt.tight_layout(pad=0.5)

    file_name = output_path + 'Figure 2' + '.jpeg'
    # plt.savefig(file_name,
    #             dpi=800,
    #             transparent=True, bbox_inches='tight', pad_inches=0, format='jpeg')
    plt.show()

