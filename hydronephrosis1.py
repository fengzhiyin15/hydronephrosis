import SimpleITK as sitk
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
import os


def setImageDetails(image1, image2):
    image1.SetOrigin(image2.GetOrigin())
    image1.SetDirection(image2.GetDirection())
    image1.SetSpacing(image2.GetSpacing())


def multiOtsu(stone_image, sep, classNum, iteNum):
    i = 0
    mask_image = sitk.Mask(stone_image, sep)
    sep_array = sitk.GetArrayFromImage(sep)
    s = tuple(range(sep_array.ndim)[::-1])
    sep_array = sep_array.transpose(s)
    sep_array_one = np.argwhere(sep_array == 1)

    while (i < iteNum):
        sep_array_one_old = sep_array_one
        otsu_multi = sitk.OtsuMultipleThresholdsImageFilter()
        otsu_multi.SetNumberOfThresholds(classNum)  # 1的时候确实是做了二分类，但要注意mask后的图像有很多0值，所以二分类时只是将0与mask的值分开，没有意义
        sep = otsu_multi.Execute(mask_image)
        sep = sep >= classNum  # 现在是取结石
        sep_array = sitk.GetArrayFromImage(sep)
        sep_array = sep_array.transpose(s)
        sep_array_one = np.argwhere(sep_array == 1)
        i = i + 1
        mask_image = sitk.Mask(stone_image, sep)
        if len(sep_array_one) == len(sep_array_one_old):
            return sep_array_one, i
    return sep_array_one, i


def tract_length_cal(body_shape, center):
    body_shape_array = sitk.GetArrayFromImage(body_shape)
    width = body_shape_array.shape[1]
    for i in range(width):
        if body_shape_array[center[2], center[1] - i, center[0] + i] == 0:
            l45 = i * np.sqrt(2)
            break
    for i in range(width):
        if body_shape_array[center[2], center[1], center[0] + i] == 0:
            l0 = i
            break
    for i in range(width):
        if body_shape_array[center[2], center[1] - i, center[0]] == 0:
            l90 = i
            break
    return (l0 + l45 + l90) // 3


data_prefix = "E:/data/hydronephrosis/one/"
num_list = os.listdir(data_prefix)
num_list.sort()
num_list = ["A002163502imp"]

cols = ["number", "stone_num", "region", "max", "volume", "other", "calyces", "uml", "sinus_box", "center",
        "stone_size"]
aa = pd.DataFrame(data=[], columns=cols)
for u in num_list:
    # for renal sinus
    images = sitk.ReadImage(os.path.join(data_prefix, u, "images.nii.gz"))
    labels = sitk.ReadImage(os.path.join(data_prefix, u, "labels_pred.nii.gz"))
    sinus_labels = labels > 1
    sinus_labels = sitk.BinaryDilate(sinus_labels, 3)
    sinus_array = sitk.GetArrayFromImage(sinus_labels)
    label_array = sitk.GetArrayFromImage(labels)
    label_array = label_array + sinus_array
    labels = sitk.GetImageFromArray(label_array)
    sinus_labels = labels > 1
    # sinus_labels = sitk.BinaryMorphologicalClosing(sinus_labels, 2)
    setImageDetails(sinus_labels, images)

    intensity_stats = sitk.LabelStatisticsImageFilter()
    intensity_stats.Execute(images, sinus_labels)
    sinus_box = intensity_stats.GetBoundingBox(1)
    sitk.WriteImage(sinus_labels, os.path.join(data_prefix, u, "sinus_labels.nii.gz"))

    # sinus part
    lower = sinus_box[4]
    upper = sinus_box[5]
    inter = (upper - lower) // 3
    sinus_array = sitk.GetArrayFromImage(sinus_labels)
    sinus_array[lower: lower + inter + 1][sinus_array[lower: lower + inter + 1] > 0] = 1
    sinus_array[lower + inter + 1:lower + (inter + 1) * 2][
        sinus_array[lower + inter + 1:lower + (inter + 1) * 2] > 0] = 2
    sinus_array[lower + (inter + 1) * 2: upper][sinus_array[lower + (inter + 1) * 2: upper] > 0] = 3
    sinus_labels_part = sitk.GetImageFromArray(sinus_array)
    sitk.WriteImage(sinus_labels_part, os.path.join(data_prefix, u, "sinus_labels_part.nii.gz"))

    # stone
    stone_image = sitk.Mask(images, sinus_labels)
    sep = stone_image > 200
    mask_image = sitk.Mask(stone_image, sep)
    mask_image_array = sitk.GetArrayFromImage(mask_image)
    # sep_array = sitk.GetArrayFromImage(sep)
    # sep_array_one = np.argwhere(sep_array == 1)
    mask_image_array1 = mask_image_array.ravel()
    order = np.argsort(-mask_image_array1)
    l = mask_image_array.shape[2]
    w = mask_image_array.shape[1]
    h = mask_image_array.shape[0]
    seg_thresholds = labels > 100
    for j in range(len(order)):
        indexOfArray = tuple((order[j] // l // w, order[j] // l % w, order[j] % l))
        if mask_image_array[indexOfArray] > 0:
            if seg_thresholds[int(indexOfArray[2]), int(indexOfArray[1]), int(indexOfArray[0])] == 0:
                seg_thresholds1 = sitk.ConnectedThreshold(images,
                                                         seedList=eval(str([(int(indexOfArray[2]), int(indexOfArray[1]), int(indexOfArray[0]))])),
                                                         lower=max(int(mask_image_array[indexOfArray]//2), 200), upper=min(int(mask_image_array[indexOfArray], 3000)))
                seg_thresholds = seg_thresholds+seg_thresholds1
        else:
            break
    # sep_array_one, itei = multiOtsu(stone_image, sep, 2, 0)
    # seedList = []
    # for i in sep_array_one:
    #     seedList.append(tuple(i)
    # # print("seedList:", seedList)
    # seedTuple = tuple(seedList)
    # # print("itei:", itei)
    # seg_thresholds = sitk.ConnectedThreshold(images, seedList=eval(str(seedTuple)), lower=200,
    #                                          upper=2000)  # 现在测量体积的最低阈值是300

    stone_label = sitk.ConnectedComponent(seg_thresholds)
    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.ComputeOrientedBoundingBoxOn()
    shape_stats.Execute(stone_label)
    setImageDetails(stone_label, images)
    sitk.WriteImage(stone_label, os.path.join(data_prefix, u, "stone.nii.gz"))

    # body shape
    h = images.GetHeight()
    w = images.GetWidth()
    spacing = images.GetSpacing()
    background = sitk.ConnectedThreshold(images, seedList=[(0, 0, 0), (0, h, 0), (w, 0, 0), (w, h, 0)], lower=-4000,
                                         upper=-200)
    body_shape = sitk.ConnectedThreshold(background, seedList=[(256, 192, 0)], lower=0, upper=0)

    # columns
    intensity_stats = sitk.LabelStatisticsImageFilter()
    intensity_stats.Execute(images, stone_label)
    ll = [l for l in intensity_stats.GetLabels() if l != 0]
    lists = []
    for i in ll:
        stats_list = [u, i, intensity_stats.GetBoundingBox(i), intensity_stats.GetMaximum(i),
                      round(shape_stats.GetPhysicalSize(i), 0), 1, 1, "", sinus_box]
        center = intensity_stats.GetBoundingBox(i)
        center = ((center[0] + center[1]) // 2, (center[2] + center[3]) // 2, (center[4] + center[5]) // 2)
        stats_list.append(center)
        stone_label_i = stone_label == i
        stone_label_ia = sitk.GetArrayFromImage(stone_label_i)
        area_each = np.sum(stone_label_ia, axis=(1, 2))
        ret = np.argmax(area_each, axis=0)
        max_area = stone_label_ia[ret]
        label_img = label(max_area)
        regions = regionprops(label_img)
        for props in regions:
            a = round(props.major_axis_length * spacing[0], 0)
            b = round(props.minor_axis_length * spacing[0], 0)
            size = round(a * b, 0)
            tract_length = round(tract_length_cal(body_shape, center) * spacing[0], 0)
        stats_list.append((ret, a, b, size, tract_length))
        lists.append(stats_list)
        # print(stats_list)
    stats1 = pd.DataFrame(data=lists, index=ll, columns=cols)
    aa = pd.concat([aa, stats1])

for i in range(len(aa)):
    upper = aa.iloc[i][cols[8]][5]
    lower = aa.iloc[i][cols[8]][4]
    inter = (upper - lower) // 3
    upp = aa.iloc[i][cols[2]][5]
    low = aa.iloc[i][cols[2]][4]
    d = upp - low
    if upp <= lower + inter:
        aa.iloc[i, 7] = "upper"
        aa.iloc[i, 6] = 1
    elif upp <= lower + inter * 2:
        if d > inter * 2 / 3:
            aa.iloc[i, 7] = "upper-medium"
            aa.iloc[i, 6] = 12
        else:
            aa.iloc[i, 7] = "medium"
            aa.iloc[i, 6] = 2
    else:
        if d > inter * 5 / 3:
            aa.iloc[i, 7] = "upper-medium-lower"
            aa.iloc[i, 6] = 123
        elif d > inter * 2 / 3:
            aa.iloc[i, 7] = "medium-lower"
            aa.iloc[i, 6] = 23
        else:
            aa.iloc[i, 7] = "lower"
            aa.iloc[i, 6] = 3

aa.to_csv("E:/data/test.csv")
