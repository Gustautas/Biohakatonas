import imgaug.augmenters as iaa
import cv2
import xml.etree.ElementTree as ET
import math
import numpy as np
import os

rotation_degrees = 270
rotation_degrees_rad = rotation_degrees / 180 * math.pi
height = 512
width = 512

images_list = []

for file in os.listdir("labeled_real_data_non_augmented_tik_simo"):
    if file.endswith(".jpg"):
        images_list.append(file[:-4])

for image in images_list:
    filename = image
    filename_with_extension = filename + ".xml"

    label_org_xml = ET.parse("labeled_real_data_non_augmented_tik_simo/" + filename_with_extension)

    annotation = label_org_xml.getroot()

    for child in annotation.findall("object"):
        bndbox = child.find("bndbox")

        xmin = bndbox.find("xmin")
        xmin_text_old = xmin.text

        xmax = bndbox.find("xmax")
        xmax_text_old = xmax.text

        ymin = bndbox.find("ymin")
        ymin_text_old = ymin.text

        ymax = bndbox.find("ymax")
        ymax_text_old = ymax.text


        xy = np.array([int(xmin_text_old) - width / 2,int(ymin_text_old) - height / 2])
        c ,s = np.cos(rotation_degrees_rad), np.sin(rotation_degrees_rad)
        R = np.array(((c, -s), (s, c)))
        rot = np.dot(R,xy)

        xmin.text = str(int(int(rot[0]) + width / 2))
        ymin.text = str(int(int(rot[1]) + height / 2))

        xy = np.array([int(xmax_text_old) - width / 2, int(ymax_text_old) - height / 2])
        c, s = np.cos(rotation_degrees_rad), np.sin(rotation_degrees_rad)
        R = np.array(((c, -s), (s, c)))
        rot = np.dot(R, xy)

        xmax.text = str(int(int(rot[0]) + width / 2))
        ymax.text = str(int(int(rot[1]) + height / 2))


        print(xmin.text)


    label_org_xml.write("labeled_real_data_augmented_tik_simo/" + filename + str(rotation_degrees) + ".xml")

    seq = iaa.Sequential([
        iaa.Rotate(rotation_degrees),
    ])

    imglist = []

    img = cv2.imread("labeled_real_data_non_augmented_tik_simo/" + filename + ".jpg")

    imglist.append(img)

    images_aug = seq.augment_images(imglist)

    cv2.imwrite("labeled_real_data_augmented_tik_simo/" + filename + str(rotation_degrees) + ".jpg", images_aug[0])
