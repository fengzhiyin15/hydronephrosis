import SimpleITK as sitk
import numpy as np
import os

data_prefix = "E:/data/hydronephrosisImages/one"
u = "A002204773"
labels = sitk.ReadImage(os.path.join(data_prefix, u, "labels.nii.gz"))
labels_pred = sitk.ReadImage(os.path.join(data_prefix, u, "labels_pred.nii.gz"))
labels = labels > 0
labels_pred = labels_pred > 0
aa=sitk.HausdorffDistanceImageFilter()
aa.Execute(labels, labels_pred)
s=aa.GetHausdorffDistance()
ss=aa.GetAverageHausdorffDistance()
print(s)
print(ss)



# labels = labels > 0
# labels_contour = sitk.BinaryContour(labels)
# labels_contour_a = sitk.GetArrayFromImage(labels_contour)
# labels_pixel = np.where(labels_contour_a != 0)
#
# labels_pred = labels_pred > 0
# labels_pred_contour = sitk.BinaryContour(labels_pred)
# labels_pred_contour_a = sitk.GetArrayFromImage(labels_pred_contour)
# labels_pred_pixel = np.where(labels_pred_contour_a != 0)
#
# a = len(labels_pixel[0])
# b = len(labels_pred_pixel[0])
# c = len(labels_pixel)
# print(a,b)
# contourDist = np.zeros((a, b))
# for i in range(a):
#     for j in range(b):
#         for k in range(c):
#             contourDist[i, j] = contourDist[i, j] + np.power(labels_pixel[k][i] - labels_pred_pixel[k][j], 2)
#
# dist = (np.sum(np.amin(contourDist, axis=0)) + np.sum(np.amin(contourDist, axis=1))) / (a + b)
# print(contourDist)
# print(dist)
