# -*- coding: utf-8 -*-
# @Time    : 2021-12-09 2:12 p.m.
# @Author  : young wang
# @FileName: contrast_adjust.py
# @Software: PyCharm
"""read volume array and adjust the conrast, save it to DICOM
"""
import os
from scipy import ndimage
from skimage import exposure
import numpy as np
from pydicom.uid import generate_uid
import pydicom
from os.path import join, isfile
import matplotlib.pyplot as plt
import glob


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
        ds.RelativeOpacity = 0.1
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
        ds
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

    desktop_loc = os.path.expanduser('~/Desktop/GeoCorrection')
    patient_uid = 'patient'
    # study_uid = 'patient'
    # input_dir = join(desktop_loc, patient_uid, study_uid)

    study_uid = 'patient'
    input_dir = join(desktop_loc, patient_uid)
    file_extension = '*.npy'

    data_path = []
    for filename in glob.glob(join(input_dir, file_extension)):
        data_path.append(filename)

    volume_data = np.load(data_path[-1])

    # remove speckles with median filter
    # volume_data = ndimage.median_filter(volume_data, size=3)

    #index is obtianed asssuming the bottom layer as 0,
    # if index = 250, it means top_remove: 80 (330-250) slices are removed from
    # the top
    top_remove = 200
    index = int(330-top_remove)

    top_stack = volume_data[:, :, index::].astype('float64')
    bottom_stack = volume_data[:, :, 0:index].astype('float64')

    opacity_factor = 0.4
    top_stack *= opacity_factor

    top_stack = top_stack.astype('uint16')
    volume_data[:, :, index::] = top_stack

    scale_factor = 466/np.max(bottom_stack)
    bottom_stack *= scale_factor

    gamma_factor = np.log(466)/np.log(np.max(bottom_stack))
    gamma_factor = np.around(gamma_factor, 2)
    print(np.max(bottom_stack))
    bottom_stack = exposure.adjust_gamma(bottom_stack, gamma_factor)
    print(np.max(bottom_stack))

    #
    desktop_loc = os.path.expanduser('~/Desktop/GeoCorrection')
    operation_uid = 'alpha blending'
    dst_uid = 'DICOM'

    input_dir = join(desktop_loc, operation_uid, study_uid,dst_uid)
    try:
        os.makedirs(input_dir)
    except FileExistsError:
        # directory already exists
        pass

    dicom_prefix = 'ci'

    PatientName = 'Test'
    top_stack = top_stack.astype('uint16')
    bottom_stack = bottom_stack.astype('uint16')

    volume_data[:, :, index::] = top_stack
    volume_data[:, :, 0:index] = bottom_stack
    # volume_data[volume_data >= 466] = 0
    # volume_data = volume_data.astype('uint16')
    # checked against the test phantom
    resolutionx, resolutiony = 0.033, 0.033
    oct_to_dicom(volume_data, resolutionx=resolutionx,
                 # resolutiony=resolutiony,
                 # PatientName=patient_uid,
                 # seriesdescription=study_uid,
                 # dicom_folder=input_dir,
                 # dicom_prefix=dicom_prefix)
                 resolutiony=resolutiony,
                 PatientName=PatientName,
                 seriesdescription=study_uid,
                 dicom_folder=input_dir,
                 dicom_prefix=dicom_prefix)