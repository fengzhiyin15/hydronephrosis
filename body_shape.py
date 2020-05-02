import SimpleITK as sitk
import numpy as np
import os

data_prefix = "E:/data/hydronephrosis/one/"
num_list = os.listdir(data_prefix)
num_list.sort()
num_list = num_list[:2]

def setImageDetails(image1, image2):
    image1.SetOrigin(image2.GetOrigin())
    image1.SetDirection(image2.GetDirection())
    image1.SetSpacing(image2.GetSpacing())

for u in num_list:
    images = sitk.ReadImage(os.path.join(data_prefix, u, "images.nii.gz"))
    labels = sitk.ReadImage(os.path.join(data_prefix, u, "labels_pred.nii.gz"))
    # h = images.GetHeight()
    # w = images.GetWidth()
    # background = sitk.ConnectedThreshold(images, seedList=[(0, 0, 0), (0, h, 0), (w, 0, 0), (w, h, 0)], lower=-4000,
    #                                      upper=-200)
    # body_shape = sitk.ConnectedThreshold(background, seedList=[(256, 192, 0)], lower=0, upper=0)

    sinus_labels = labels > 1
    sinus_labels = sitk.BinaryDilate(sinus_labels, 3)
    sinus_array = sitk.GetArrayFromImage(sinus_labels)
    label_array = sitk.GetArrayFromImage(labels)
    label_array = label_array + sinus_array
    labels = sitk.GetImageFromArray(label_array)
    sinus_labels = labels > 1
    sinus_labels = sitk.BinaryMorphologicalClosing(sinus_labels, 2)
    setImageDetails(sinus_labels, images)

    intensity_stats = sitk.LabelStatisticsImageFilter()
    intensity_stats.Execute(images, sinus_labels)
    s = intensity_stats.GetBoundingBox(1)
    print(s)

    sitk.WriteImage(sinus_labels, os.path.join(data_prefix, u, "sinus_labels.nii.gz"))

h = images.GetHeight()
w = images.GetWidth()
background = sitk.ConnectedThreshold(images, seedList=[(0, 0, 0), (0, h, 0), (w, 0, 0), (w, h, 0)], lower=-4000,
                                     upper=-200)
body_shape = sitk.ConnectedThreshold(background, seedList=[(256, 192, 0)], lower=0, upper=0)
def tract_length(body_shape, center):
    body_shape_array=sitk.GetArrayFromImage(body_shape)
    width=body_shape_array.shape[1]
    for i in range(width):
        if body_shape_array[center[2], center[1]-i, center[0]+i] == 0:
            l45=i*np.sqrt(2)
            break
    for i in range(width):
        if body_shape_array[center[2], center[1], center[0]+i]==0:
            l0 = i
            break
    for i in range(width):
        if body_shape_array[center[2], center[1]-i, center[0]]==0:
            l90 = i
            break
    return (l0+l45+l90)//3



