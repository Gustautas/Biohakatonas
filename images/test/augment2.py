import imgaug.augmenters as iaa
import cv2
import xml.etree.ElementTree as ET
import math
import numpy as np

filename = "skaiciaitest"
filename_with_extension = filename + ".xml"

height = 512
width = 512

label_org_xml = ET.parse(filename_with_extension)

annotation = label_org_xml.getroot()

rotation_degrees = 180
rotation_degrees_rad = rotation_degrees / 180 * math.pi
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

    xmin.text = str(int(rot[0]) + width / 2)
    ymin.text = str(int(rot[1]) + height / 2)

    xy = np.array([int(xmax_text_old) - width / 2, int(ymax_text_old) - height / 2])
    c, s = np.cos(rotation_degrees_rad), np.sin(rotation_degrees_rad)
    R = np.array(((c, -s), (s, c)))
    rot = np.dot(R, xy)

    xmax.text = str(int(rot[0]) + width / 2)
    ymax.text = str(int(rot[1]) + height / 2)
    print("XMIN", xmin.text)
    print("XMAX", xmax.text)
    print("YMIN", ymin.text)
    print("YMAX", ymax.text)

print("ROT DEG", rotation_degrees)
label_org_xml.write(filename + str(rotation_degrees) + ".xml")

seq = iaa.Sequential([
    iaa.Rotate(rotation_degrees),
])

imglist = []

img = cv2.imread(filename + ".png")

imglist.append(img)

images_aug = seq.augment_images(imglist)

cv2.imwrite(filename + str(rotation_degrees) + ".jpg", images_aug[0])