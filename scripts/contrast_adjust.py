# -*- coding: utf-8 -*-
# @Time    : 2021-12-09 2:12 p.m.
# @Author  : young wang
# @FileName: contrast_adjust.py
# @Software: PyCharm
"""read volume array and adjust the conrast, save it to DICOM
"""

from skimage import data, exposure, img_as_float
import numpy as np
from pydicom.uid import generate_uid
import pydicom
from os.path import join, isfile
import matplotlib.pyplot as plt


def oct_to_dicom(data, resolutionx, resolutiony, PatientName, seriesdescription,
                 dicom_folder, dicom_prefix):
    """
    convert pixel array [512,512,330] to DICOM format
    using MRI template, this template will be deprecated
    in the future once OCT template is received
    """
    # data = load_from_oct_file(oct_file)

    dss = []

    template_file = '../template data/template.dcm'
    ds = pydicom.dcmread(template_file)

    # SeriesInstanceUID refers to each series, and should be
    # unqie for each sesssion, and generate_uid() provides an unique
    # identifier
    ds.SeriesInstanceUID = generate_uid()

    all_files_exist = False

    # looping through all 330 slice of images with [512(row) x 512(column)]
    for i in range(data.shape[2]):
        # UID used for indexing slices
        ds.SOPInstanceUID = generate_uid()

        # update row and column numbers to be 512
        ds.Rows = data.shape[0]
        ds.Columns = data.shape[1]

        # define the bottom(assuming the middle plane to be zero,
        # that -165 * 30um(axial resolution) = -4.95 mm)
        # DICOM assumes physical dimension to be in mm unit
        bottom = -4.95

        # elevate the z by its axial resolution at a time
        z = bottom + (i * 0.03)

        # update meta properties

        # 1cm / 512 = 0.02 mm, needs to check with rob
        # this spacing should be calculated as radiant/pixel then mm to pixel
        #
        ds.PixelSpacing = [resolutionx, resolutiony]  # pixel spacing in x, y planes [mm]
        ds.SliceThickness = 0.03  # slice thickness in axial(z) direction [mm]
        ds.SpacingBetweenSlices = 0.03  # slice spacing in axial(z) direction [mm]
        ds.SliceLocation = '%0.2f' % z  # slice location in axial(z) direction
        ds.InstanceNumber = '%0d' % (i + 1,)  # instance number, 330 in total
        ds.ImagePositionPatient = [z, 0, 0]  # patient physical location
        ds.Manufacturer = 'Audioptics Medical Inc'
        ds.InstitutionName = 'Audioptics Medical'
        ds.InstitutionAddress = '1344 Summer St., #55, Halifax, NS, Canada'
        ds.StudyDescription = 'Example DICOM export'
        ds.StationName = 'Unit 1'
        ds.SeriesDescription = seriesdescription
        ds.PhysiciansOfRecord = ''
        ds.PerformingPhysicianName = ''
        ds.InstitutionalDepartmentName = ''
        ds.ManufacturerModelName = 'Mark II'
        ds.PatientName = PatientName
        ds.PatientBirthDate = '20201123'
        ds.PatientAddress = ''

        # setting the dynamic range with WindowCenter and WindowWidth
        # lowest_visible_value = window_center — window_width / 2
        # highest_visible_value = window_center + window_width / 2

        ds.WindowCenter = '248'
        ds.WindowWidth = '396'

        # # set highest and lowest pixel values
        ds.LargestImagePixelValue = 446
        ds.SmallestImagePixelValue = 50
        dicom_file = join(dicom_folder, "%s%05d.dcm" % (dicom_prefix, i))

        pixel_data = data[:, :, i]
        # pixel_data[pixel_data <= 50] = 0
        ds.PixelData = pixel_data.tobytes()
        ds.save_as(dicom_file)
        dss.append(ds)
        all_files_exist = all_files_exist and isfile(dicom_file)
    return all_files_exist

''
if __name__ == '__main__':
    
    data = np.load('/Users/youngwang/Desktop/GeoCorrection/cadaver/partial insertion I/ci-cadaver.npy')
    data =data.astype('float64')
    data = data/data.max()
    # new_data = exposure.equalize_adapthist(data, clip_limit=0.7)*466
    # clip = 0.1
    #
    # p2, p98 = np.percentile(np.ravel(data), (clip, 100-clip))
    # new_data = exposure.rescale_intensity(data, in_range=(p2, p98))*466

    new_data = exposure.adjust_gamma(data, 0.2)

    new_data = 466*new_data/new_data.max()

    new_data = new_data.astype(np.uint16)

    directory = '/Users/youngwang/Desktop/GeoCorrection/equalization'


    # dicom_prefix = 'CI-cadaver'
    dicom_prefix = 'equ'

    seriesdescription = ['partial insertion I']


    export_path = directory+'/DICOM'
    from os import path
    import os
    try:
        os.makedirs(export_path)
    except FileExistsError:
        # directory already exists
        pass


    PatientName = 'EQU'

    # checked against the test phantom
    resolutionx, resolutiony = 0.033, 0.033
    oct_to_dicom(new_data, resolutionx=resolutionx,
                 resolutiony=resolutiony,
                 PatientName=PatientName,
                 seriesdescription=seriesdescription[0],
                 dicom_folder=export_path,
                 dicom_prefix=dicom_prefix)

    fig,ax = plt.subplots(1,2)
    ax[0].hist(np.ravel(data))
    ax[1].hist(np.ravel(new_data))
    plt.show()

    # npy_name = PatientName+'.npy'
    # npy_path = join(directory,npy_name)
    # with open(npy_path, 'wb') as f:
    #     np.save(f, data)
